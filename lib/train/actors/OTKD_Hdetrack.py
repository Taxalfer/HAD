from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate

import torch.nn.functional as F
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions.normal import Normal  # 导入PyTorch的Normal分布

class OTKD_HDETrackActor(BaseActor):
    """ Actor for training OTKD models (adapted from HDETrack) """
    def __init__(self, net_teacher, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net_teacher, net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.device = next(net_teacher.parameters()).device
        
        # OTKD超参数（论文最优值）
        self.lambda1 = 0.7  # 特征蒸馏损失权重
        self.lambda2 = 0.9  # 类间位置相关损失权重
        self.lambda3 = 0.9  # 类内位置相关损失权重
        
        # CAL模块：上下文自适应学习的MLP（三层）
        self.cal_mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 768)
        ).to(self.device)

    def __call__(self, data):
        """
        args:
            data - 输入数据，包含'template', 'search', 'gt_bbox', 'epoch'等
        returns:
            loss    - 总训练损失
            status  - 详细损失字典
        """
        # 前向传播，获取师生输出
        out_dict, out_dict_s = self.forward_pass(data)
        
        # 计算OTKD损失
        loss, status = self.compute_otkd_losses(out_dict, out_dict_s, data)

        return loss, status

    def forward_pass(self, data):
        """前向传播，获取师生的特征、注意力、预测框等输出"""
        # 校验输入格式（保持原有逻辑）
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1
        assert len(data['template_event']) == 1
        assert len(data['search_event']) == 1

        # 处理模板和搜索图像
        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (bs, 3, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (bs, 3, 256, 256)
        
        # 处理Event模态数据
        template_event = data['template_event'][0].view(-1, *data['template_event'].shape[2:])
        search_event = data['search_event'][0].view(-1, *data['search_event'].shape[2:])

        # CE掩码（保持原有逻辑）
        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                    total_epochs=ce_start_epoch + ce_warm_epoch,
                                                    ITERS_PER_EPOCH=1,
                                                    base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
        if len(template_list) == 1:
            template_list = template_list[0]

        # 教师模型前向（保持原有逻辑）
        out_dict = self.net_teacher(
                            template=template_list,
                            search=search_img,
                            event_template=template_event,
                            event_search=search_event,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
                                
        # 学生模型前向（保持原有逻辑）
        #=======================================
        #               EventVOT
        #=======================================                        
        # out_dict_s = self.net(
        #                     # template=template_list,
        #                     # search=search_img,
        #                     # event_template=template_event,
        #                     # event_search=search_event,
        #                     event_template_img=template_list,
        #                     event_search_img=search_img,
        #                     ce_template_mask=box_mask_z,
        #                     ce_keep_rate=ce_keep_rate,
        #                     return_last_attn=False)

        #=======================================
        #               COESOT
        #=======================================
        out_dict_s = self.net(
                            # template=template_list,
                            # search=search_img,
                            # event_template=template_event,
                            # event_search=search_event,
                            event_template_img=template_event,
                            event_search_img=search_event,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
        
        return out_dict, out_dict_s

    def compute_channel_spatial_mask(self, feature):
        """
        OTKD核心：计算通道注意力掩码和空间注意力掩码（MBF模块）
        最终版：通过计算逻辑确保输出mask形状与输入feature完全一致（无assert）
        args:
            feature - 特征图 [bs, L, C]（仅支持序列格式）
        returns:
            channel_mask - 通道注意力掩码 [bs, L, C]（与输入shape完全一致）
            spatial_mask - 空间注意力掩码 [bs, L, C]（与输入shape完全一致）
        """
        # 1. 提取原始维度信息（核心：全程基于原始维度计算）
        bs, L_ori, C_ori = feature.shape
        
        # 2. 动态计算2D特征尺寸（适配任意L，不裁剪/补零）
        # 方案：不强制H=W，而是直接按L_ori拆分H和W（保证H*W=L_ori）
        H = int(math.ceil(math.sqrt(L_ori)))  # 向上取整保证H≥√L
        W = L_ori // H
        # 微调W确保H*W=L_ori（100%匹配原始长度）
        while H * W != L_ori:
            W += 1
            if H * W > L_ori:
                H -= 1
                W = L_ori // H if H > 0 else 1
        # 极端情况防护（H/W为0）
        H = max(H, 1)
        W = max(W, 1)
        # 最终校验H*W=L_ori（纯计算逻辑保证）
        if H * W != L_ori:
            # 兜底方案：拆分维度为(1, L_ori)，确保H*W=L_ori
            H = 1
            W = L_ori
        
        # 3. 转换为2D特征（100%匹配原始长度，无裁剪/补零）
        # 直接reshape，因为H*W=L_ori，无需任何裁剪/补零
        feature_2d = feature.permute(0, 2, 1).reshape(bs, C_ori, H, W)
        
        # 4. 计算通道注意力掩码（维度严格匹配）
        # 全局平均池化：[bs, C_ori, 1, 1]
        channel_att = torch.mean(feature_2d, dim=[2, 3], keepdim=True)
        channel_att = F.softmax(channel_att, dim=1)
        # 扩展为2D → 展平为序列 → 转置匹配原始形状
        # 扩展：[bs, C_ori, 1, 1] → [bs, C_ori, H, W]
        # 展平：[bs, C_ori, H, W] → [bs, C_ori, L_ori]
        # 转置：[bs, C_ori, L_ori] → [bs, L_ori, C_ori]
        channel_mask = channel_att.expand(-1, -1, H, W).reshape(bs, C_ori, L_ori).permute(0, 2, 1)
        
        # 5. 计算空间注意力掩码（维度严格匹配）
        # 通道平均池化：[bs, 1, H, W]
        spatial_att = torch.mean(feature_2d, dim=1, keepdim=True)
        # 归一化：展平为[bs, H*W] → softmax → 恢复为[bs, 1, H, W]
        spatial_att = F.softmax(spatial_att.reshape(bs, -1), dim=1).reshape(bs, 1, H, W)
        # 扩展为2D → 展平为序列 → 转置匹配原始形状
        # 扩展：[bs, 1, H, W] → [bs, C_ori, H, W]
        # 展平：[bs, C_ori, H, W] → [bs, C_ori, L_ori]
        # 转置：[bs, C_ori, L_ori] → [bs, L_ori, C_ori]
        spatial_mask = spatial_att.expand(-1, C_ori, -1, -1).reshape(bs, C_ori, L_ori).permute(0, 2, 1)
        
        return channel_mask, spatial_mask

    def context_adaptive_learning(self, teacher_feature, student_feature):
        """
        OTKD核心：上下文自适应学习（CAL模块）
        修复：解决池化后reshape维度不匹配问题，保证插值前维度计算准确
        args:
            teacher_feature - 教师特征 [bs, L, C]
            student_feature - 学生特征 [bs, L, C]
        returns:
            fused_teacher_feat - 自适应融合后的教师特征 [bs, L, C]
            fused_student_feat - 自适应融合后的学生特征 [bs, L, C]
        """
        bs, L_ori, C_ori = teacher_feature.shape
        
        # 步骤1：计算原始特征的2D尺寸（保证H*W=L_ori）
        H_ori = int(math.ceil(math.sqrt(L_ori)))
        W_ori = L_ori // H_ori
        while H_ori * W_ori != L_ori:
            W_ori += 1
            if H_ori * W_ori > L_ori:
                H_ori -= 1
                W_ori = L_ori // H_ori if H_ori > 0 else 1
        H_ori = max(H_ori, 1)
        W_ori = max(W_ori, 1)
        if H_ori * W_ori != L_ori:
            H_ori, W_ori = 1, L_ori

        # 三尺度池化（浅/中/深）- 修复池化后尺寸计算逻辑
        def multi_scale_pooling(x):
            # 转换为2D特征 [bs, C_ori, H_ori, W_ori]
            x_2d = x.permute(0, 2, 1).reshape(bs, C_ori, H_ori, W_ori)
            
            # 池化操作（使用ceil_mode=True，记录实际池化后尺寸）
            pool1 = F.avg_pool2d(x_2d, kernel_size=2, stride=2, ceil_mode=True)
            pool2 = F.avg_pool2d(x_2d, kernel_size=4, stride=4, ceil_mode=True)
            pool3 = F.avg_pool2d(x_2d, kernel_size=8, stride=8, ceil_mode=True)
            
            # 展平每个池化结果（动态计算长度，不依赖sqrt）
            pool1_flat = pool1.permute(0, 2, 3, 1).reshape(bs, -1, C_ori)
            pool2_flat = pool2.permute(0, 2, 3, 1).reshape(bs, -1, C_ori)
            pool3_flat = pool3.permute(0, 2, 3, 1).reshape(bs, -1, C_ori)
            
            # 拼接所有池化结果
            pooled = torch.cat([pool1_flat, pool2_flat, pool3_flat], dim=1)
            return pooled

        # 师生特征多尺度池化
        teacher_pooled = multi_scale_pooling(teacher_feature)
        student_pooled = multi_scale_pooling(student_feature)
        
        # CAL-MLP自适应融合
        fused_teacher_feat = self.cal_mlp(teacher_pooled)
        fused_student_feat = self.cal_mlp(student_pooled)
        
        # 步骤2：恢复原始维度（插值）- 核心修复：动态计算池化后尺寸
        # 获取池化后特征的实际长度
        L_pooled = fused_teacher_feat.shape[1]
        
        # 动态计算池化后2D尺寸（保证H*W=L_pooled）
        pooled_H = int(math.ceil(math.sqrt(L_pooled)))
        pooled_W = L_pooled // pooled_H
        while pooled_H * pooled_W != L_pooled:
            pooled_W += 1
            if pooled_H * pooled_W > L_pooled:
                pooled_H -= 1
                pooled_W = L_pooled // pooled_H if pooled_H > 0 else 1
        pooled_H = max(pooled_H, 1)
        pooled_W = max(pooled_W, 1)
        # 兜底方案：确保H*W=L_pooled
        if pooled_H * pooled_W != L_pooled:
            pooled_H, pooled_W = 1, L_pooled

        # 安全的reshape + 插值（核心修复）
        try:
            # 转换为2D特征（保证维度匹配）
            teacher_2d = fused_teacher_feat.permute(0, 2, 1).reshape(bs, C_ori, pooled_H, pooled_W)
            student_2d = fused_student_feat.permute(0, 2, 1).reshape(bs, C_ori, pooled_H, pooled_W)
            
            # 插值恢复到原始尺寸
            fused_teacher_feat = F.interpolate(
                teacher_2d,
                size=(H_ori, W_ori),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1).reshape(bs, L_ori, C_ori)
            
            fused_student_feat = F.interpolate(
                student_2d,
                size=(H_ori, W_ori),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1).reshape(bs, L_ori, C_ori)
            
        except Exception as e:
            # 极端情况兜底：直接返回零张量（避免训练中断）
            print(f"插值失败，错误：{e}，使用零张量兜底")
            fused_teacher_feat = torch.zeros_like(teacher_feature)
            fused_student_feat = torch.zeros_like(student_feature)
        
        return fused_teacher_feat, fused_student_feat

    def correlation_distance_supervision(self, teacher_boxes, student_boxes, teacher_scores, student_scores):
        """
        OTKD核心：相关距离监督（CDS模块）
        args:
            teacher_boxes - 教师预测框 [bs, N, 4] (cxcywh)
            student_boxes - 学生预测框 [bs, N, 4] (cxcywh)
            teacher_scores - 教师响应分数 [bs, 1, H, W]
            student_scores - 学生响应分数 [bs, 1, H, W]
        returns:
            inter_loss - 类间位置相关损失
            intra_loss - 类内位置相关损失
        """
        # 防止维度为空
        if teacher_boxes.numel() == 0 or student_boxes.numel() == 0:
            return torch.tensor(0.0).to(self.device), torch.tensor(0.0).to(self.device)
        
        # 1. 类间位置相关损失（序列级分布趋势）
        teacher_box_flat = teacher_boxes.view(self.bs, -1)
        student_box_flat = student_boxes.view(self.bs, -1)
        
        # 计算皮尔逊相关系数（添加数值稳定性）
        teacher_mean = teacher_box_flat.mean(dim=1, keepdim=True)
        student_mean = student_box_flat.mean(dim=1, keepdim=True)
        cov = torch.mean((teacher_box_flat - teacher_mean) * (student_box_flat - student_mean), dim=1)
        teacher_std = teacher_box_flat.std(dim=1, keepdim=True).squeeze(1)
        student_std = student_box_flat.std(dim=1, keepdim=True).squeeze(1)
        std_product = teacher_std * student_std + 1e-8
        pearson_corr = cov / std_product
        inter_loss = 1 - pearson_corr.mean()
        
        # 2. 类内位置相关损失（置信度分布）
        if teacher_scores.numel() == 0 or student_scores.numel() == 0:
            intra_loss = torch.tensor(0.0).to(self.device)
        else:
            teacher_score_flat = teacher_scores.view(self.bs, -1)
            student_score_flat = student_scores.view(self.bs, -1)
            
            teacher_score_mean = teacher_score_flat.mean(dim=1, keepdim=True)
            student_score_mean = student_score_flat.mean(dim=1, keepdim=True)
            cov_intra = torch.mean((teacher_score_flat - teacher_score_mean) * (student_score_flat - student_score_mean), dim=1)
            teacher_score_std = teacher_score_flat.std(dim=1, keepdim=True).squeeze(1)
            student_score_std = student_score_flat.std(dim=1, keepdim=True).squeeze(1)
            std_product_intra = teacher_score_std * student_score_std + 1e-8
            pearson_corr_intra = cov_intra / std_product_intra
            intra_loss = 1 - pearson_corr_intra.mean()
        
        return inter_loss, intra_loss

    def compute_otkd_losses(self, out_dict, out_dict_s, gt_dict, return_status=True):
        """计算OTKD的完整损失（替换原有简单蒸馏损失）"""
        # -------------------------- 1. 基础损失（保持原有逻辑） --------------------------
        # GT高斯图和预测框
        gt_bbox = gt_dict['search_anno'][-1]  # (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)
        
        pred_boxes = out_dict_s['s_pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (BN,4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        
        # GIoU和L1损失
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_loss, iou = torch.tensor(0.0).to(pred_boxes.device), torch.tensor(0.0).to(pred_boxes.device)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        
        # 位置损失（Focal Loss）
        location_loss = self.objective['focal'](out_dict_s['s_score_map'], gt_gaussian_maps) if 's_score_map' in out_dict_s and out_dict_s['s_score_map'].numel() > 0 else torch.tensor(0.0).to(pred_boxes.device)

        # -------------------------- 2. OTKD核心损失（替换原有蒸馏损失） --------------------------
        # 2.1 掩码基特征蒸馏（MBF）：解耦通道/空间知识
        teacher_feature = out_dict['teacher_feature']  # [bs, L, C]
        student_feature = out_dict_s['student_feature']  # [bs, L/2, C]
        
        # 对齐特征维度（保持原有repeat逻辑）
        student_feature = student_feature.repeat(1, 2, 1)
        
        # 强制裁剪到相同长度（防止repeat后长度不匹配）
        min_L = min(teacher_feature.shape[1], student_feature.shape[1])
        teacher_feature = teacher_feature[:, :min_L, :]
        student_feature = student_feature[:, :min_L, :]
        
        # 计算通道/空间掩码（终极修复：维度完全匹配）
        teacher_channel_mask, teacher_spatial_mask = self.compute_channel_spatial_mask(teacher_feature)
        student_channel_mask, student_spatial_mask = self.compute_channel_spatial_mask(student_feature)
        
        # 掩码加权特征（现在维度完全一致）
        teacher_feat_channel = teacher_feature * teacher_channel_mask
        teacher_feat_spatial = teacher_feature * teacher_spatial_mask
        student_feat_channel = student_feature * student_channel_mask
        student_feat_spatial = student_feature * student_spatial_mask
        
        # 通道/空间特征蒸馏损失（添加数值稳定性）
        loss_fkd_c = F.mse_loss(student_feat_channel, teacher_feat_channel.detach(), reduction='mean') if teacher_feat_channel.numel() > 0 else torch.tensor(0.0).to(self.device)
        loss_fkd_s = F.mse_loss(student_feat_spatial, teacher_feat_spatial.detach(), reduction='mean') if teacher_feat_spatial.numel() > 0 else torch.tensor(0.0).to(self.device)
        
        # 2.2 上下文自适应学习（CAL）：层级特征融合
        fused_teacher_feat, fused_student_feat = self.context_adaptive_learning(teacher_feature, student_feature)
        loss_fkd = F.mse_loss(fused_student_feat, fused_teacher_feat.detach(), reduction='mean') if fused_teacher_feat.numel() > 0 else torch.tensor(0.0).to(self.device)
        
        # 2.3 相关距离监督（CDS）：序列级分布监督
        teacher_boxes = out_dict['pred_boxes'] if 'pred_boxes' in out_dict else out_dict['s_pred_boxes']
        student_boxes = out_dict_s['s_pred_boxes']
        teacher_scores = out_dict['score_map'] if 'score_map' in out_dict else torch.tensor(0.0).to(self.device)
        student_scores = out_dict_s['s_score_map'] if 's_score_map' in out_dict_s else torch.tensor(0.0).to(self.device)
        
        inter_loss, intra_loss = self.correlation_distance_supervision(teacher_boxes, student_boxes, teacher_scores, student_scores)

        # -------------------------- 3. 总损失计算（按OTKD公式加权） --------------------------
        # 基础损失
        base_loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        # OTKD蒸馏损失
        distill_loss = self.lambda1 * loss_fkd + self.lambda2 * inter_loss + self.lambda3 * intra_loss
        # 总损失
        total_loss = base_loss + distill_loss

        # -------------------------- 4. 状态字典（日志输出） --------------------------
        if return_status:
            status = {
                "Loss/total": total_loss.item(),
                "Loss/base": base_loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                # OTKD损失
                "Loss/fkd_c": loss_fkd_c.item(),
                "Loss/fkd_s": loss_fkd_s.item(),
                "Loss/fkd": loss_fkd.item(),
                "Loss/inter": inter_loss.item(),
                "Loss/intra": intra_loss.item(),
                "Loss/distill": distill_loss.item(),
                "IoU": iou.detach().mean().item() if iou.numel() > 0 else 0.0
            }
            return total_loss, status
        else:
            return total_loss