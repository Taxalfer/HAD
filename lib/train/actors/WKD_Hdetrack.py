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

class SinkhornDistance(nn.Module):
    """
    Sinkhorn算法求解离散Wasserstein距离（熵正则化版）
    适配WKD论文公式（3）：min_q ∑c_ij q_ij + η ∑q_ij log q_ij
    约束：q_ij ≥ 0, ∑j q_ij = p_i^T, ∑i q_ij = p_j^S
    """
    def __init__(self, eps=0.05, max_iter=9, stop_thresh=1e-3):
        """
        args:
            eps: 熵正则化参数（论文默认0.05）
            max_iter: 最大迭代次数（论文默认9次）
            stop_thresh: 收敛停止阈值（低于该值则提前终止）
        """
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.stop_thresh = stop_thresh
        self.epsilon = 1e-10  # 数值稳定性保护

    def forward(self, teacher_prob, student_prob, cost_matrix):
        """
        args:
            teacher_prob: 教师概率分布 [bs, num_queries_t, 1]（单目标场景为目标概率）
            student_prob: 学生概率分布 [bs, num_queries_s, 1]
            cost_matrix: 运输成本矩阵 [bs, num_queries_t, num_queries_s]（由类别关联IR转换）
        returns:
            wd_distance: 批量的Wasserstein距离 [bs]
            transport_plan: 最优运输计划 [bs, num_queries_t, num_queries_s]
            convergence: 收敛标志（是否达到停止阈值）
        """
        bs, num_t, _ = teacher_prob.shape
        num_s = student_prob.shape[1]

        # 1. 初始化运输计划的对数形式（数值稳定）
        log_q = -self.eps * cost_matrix  # [bs, num_t, num_s]
        log_q = log_q - log_q.logsumexp(dim=(-2, -1), keepdim=True)  # 归一化初始化

        # 2. Sinkhorn迭代（交替归一化行和列，满足边际约束）
        convergence = torch.ones(bs, dtype=torch.bool, device=teacher_prob.device)
        for iter_idx in range(self.max_iter):
            # 迭代1：归一化列，满足学生概率约束 ∑i q_ij = p_j^S
            log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True)  # 列求和=1
            # 迭代2：归一化行，满足教师概率约束 ∑j q_ij = p_i^T
            log_q = log_q + torch.log(teacher_prob + self.epsilon)  # 行目标=p_i^T
            log_q = log_q - torch.logsumexp(log_q, dim=2, keepdim=True)  # 行求和=1

            # 检查收敛性（运输计划变化量低于阈值则停止）
            if iter_idx > 1:
                current_plan = torch.exp(log_q)
                plan_diff = torch.abs(current_plan - prev_plan).mean(dim=(-2, -1))
                convergence = plan_diff < self.stop_thresh
                if convergence.all():
                    break
            prev_plan = torch.exp(log_q)

        # 3. 计算最终Wasserstein距离（论文公式3的目标函数值）
        transport_plan = torch.exp(log_q)  # [bs, num_t, num_s]
        # 熵正则项：η * ∑q_ij log q_ij
        entropy_term = transport_plan * (torch.log(transport_plan + self.epsilon) - math.log(self.eps))
        entropy_term = entropy_term.sum(dim=(-2, -1))  # [bs]
        # 运输成本项：∑c_ij q_ij
        cost_term = (cost_matrix * transport_plan).sum(dim=(-2, -1))  # [bs]
        # 总WD距离 = 成本项 + 熵正则项
        wd_distance = cost_term + entropy_term

        return wd_distance.mean(), transport_plan, convergence


class WKD_L_SingleObject(nn.Module):
    def __init__(self, temperature=2.0, sharpness=1.0, reg=0.05):
        super().__init__()
        self.temp = temperature
        self.sharpness = sharpness
        self.sinkhorn = SinkhornDistance(eps=reg, max_iter=9)

    def extract_target_score(self, heatmap):
        """从热力图中提取目标分数（单目标场景）"""
        bs, H, W = heatmap.shape
        # 方案1：热力图全局最大值（最直接的目标分数）
        target_score, _ = torch.max(heatmap.view(bs, -1), dim=-1)  # [bs]
        # 扩展为[bs, Q, 1]格式（Q=1，单目标无需多query）
        target_score = target_score.unsqueeze(-1).unsqueeze(-1)  # [bs, 1, 1]
        return target_score

    def compute_target_region_IR(self, teacher_heatmap):
        """基于热力图计算目标区域关联（单目标适配）"""
        bs, H, W = teacher_heatmap.shape
        # 提取目标区域（热力图Top-30%像素）
        target_thresh = torch.topk(teacher_heatmap.view(bs, -1), k=int(0.3*H*W), dim=-1)[0][:, -1].unsqueeze(-1).unsqueeze(-1)
        target_mask = (teacher_heatmap >= target_thresh).float()  # [bs, H, W]
        # 展平为序列
        target_feat_flat = teacher_heatmap.view(bs, 1, -1)  # [bs, 1, H*W]
        target_mask_flat = target_mask.view(bs, 1, -1)  # [bs, 1, H*W]
        # 计算目标区域内的相似度（IR）
        target_feat_norm = F.normalize(target_feat_flat, p=2, dim=-1)
        ir_matrix = torch.matmul(target_feat_norm.transpose(1, 2), target_feat_norm)  # [bs, H*W, H*W]
        # 只保留目标区域的关联
        ir_matrix = ir_matrix * target_mask_flat.transpose(1, 2) * target_mask_flat
        return ir_matrix

    def forward(self, teacher_heatmap, student_heatmap):
        """
        输入：仅依赖heatmap字段
        teacher_heatmap: 教师热力图 [bs, H, W]
        student_heatmap: 学生热力图 [bs, H, W]
        """
        # 1. 提取目标分数（替代score/pred_score）
        teacher_score = self.extract_target_score(teacher_heatmap)  # [bs, 1, 1]
        student_score = self.extract_target_score(student_heatmap)  # [bs, 1, 1]

        # 2. 软化概率
        teacher_prob = F.softmax(teacher_score / self.temp, dim=-1)  # [bs, 1, 1]
        student_prob = F.log_softmax(student_score / self.temp, dim=-1)  # [bs, 1, 1]

        # 3. 计算目标区域关联和成本矩阵
        ir_matrix = self.compute_target_region_IR(teacher_heatmap)  # [bs, N, N]
        bs, Q, _ = teacher_prob.shape
        # 下采样到query数量（Q=1）
        ir_matrix_downsampled = F.adaptive_avg_pool2d(ir_matrix.unsqueeze(1), (Q, Q)).squeeze(1)  # [bs, 1, 1]
        cost_matrix = 1 - torch.exp(-self.sharpness * (1 - ir_matrix_downsampled))  # [bs, 1, 1]

        # 4. 计算WD距离
        total_loss = 0.0
        for b in range(bs):
            t_prob = teacher_prob[b].unsqueeze(0)  # [1, 1, 1]
            s_prob = student_prob[b].unsqueeze(0)  # [1, 1, 1]
            cost = cost_matrix[b].unsqueeze(0)  # [1, 1, 1]
            distance, _, _ = self.sinkhorn(t_prob, s_prob, cost)
            total_loss += distance.mean()

        return total_loss / bs

# ---------------------- 3. WKD-F（适配teacher_feature字段） ----------------------
class WKD_F_SingleObject(nn.Module):
    def __init__(self, d_model=768, mean_cov_ratio=2.0, device=None):
        super().__init__()
        self.mean_cov_ratio = mean_cov_ratio
        self.device = device
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        ).to(self.device)

    def fit_gaussian(self, features, mask=None):
        """拟合高斯分布（修复掩码维度匹配）"""
        bs, seq_len, d_model = features.shape
        if mask is not None:
            # 核心修复1：强制掩码长度与特征序列长度一致
            if mask.shape[1] != seq_len:
                # 下采样/上采样掩码到特征序列长度
                mask = F.adaptive_avg_pool1d(
                    mask.unsqueeze(1),  # [bs, 1, mask_len]
                    output_size=seq_len  # 适配特征序列长度
                ).squeeze(1)  # [bs, seq_len]
                # 二值化（确保掩码仍是0/1分布）
                mask = (mask > 0.5).float()

            mask = mask.unsqueeze(-1).repeat(1, 1, d_model)  # [bs, seq_len, d_model]
            features = features * mask
            valid_count = mask.sum(dim=1, keepdim=True) + 1e-5
            mean = (features.sum(dim=1)) / valid_count.squeeze(1)  # [bs, d_model]
            var = ((features - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / valid_count.squeeze(1) + 1e-5  # [bs, d_model]
        else:
            mean = torch.mean(features, dim=1)  # [bs, d_model]
            var = torch.var(features, dim=1) + 1e-5  # [bs, d_model]
        return mean, var

    def wasserstein_gaussian(self, t_mean, t_var, s_mean, s_var):
        mean_dist = torch.norm(t_mean - s_mean, p=2, dim=-1).square()
        var_dist = torch.sum(torch.sqrt(t_var) + torch.sqrt(s_var) - 2 * torch.sqrt(torch.sqrt(t_var) * torch.sqrt(s_var)), dim=-1)
        return self.mean_cov_ratio * mean_dist + var_dist

    def forward(self, teacher_feat, student_feat, search_mask=None):
        # 设备对齐
        teacher_feat = teacher_feat.to(self.device)
        student_feat = student_feat.to(self.device)
        if search_mask is not None:
            search_mask = search_mask.to(self.device)

        # 学生特征投影
        student_feat_proj = self.projector(student_feat)

        # 拟合高斯分布（掩码已自动适配长度）
        t_mean, t_var = self.fit_gaussian(teacher_feat, mask=search_mask)
        s_mean, s_var = self.fit_gaussian(student_feat_proj, mask=search_mask)

        # 计算损失
        wd_loss = self.wasserstein_gaussian(t_mean, t_var, s_mean, s_var)
        return wd_loss.mean()


class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_size]
        out, _ = self.gru(x)
        # Take the last time step's output
        out = self.fc(out[:, -1, :])
        return out

class WKD_HDETrackActor(BaseActor):
    """ Actor for training HDETrack models """
    def __init__(self, net_teacher, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net_teacher, net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

        # 修复3：自动获取模型所在设备（从教师模型参数中提取）
        self.device = next(net_teacher.parameters()).device  # 关键：获取GPU/CPU设备
        print(f"Model running on device: {self.device}")  # 打印设备信息（可选，用于验证）

        # 初始化WKD模块（传递设备给WKD-F）
        self.wkd_l = WKD_L_SingleObject(temperature=2.0, sharpness=1.0).to(self.device)
        self.feature_dim = 768
        # 修复4：创建WKD-F时传入设备
        self.wkd_f = WKD_F_SingleObject(d_model=self.feature_dim, mean_cov_ratio=2.0, device=self.device)
    
    def generate_search_mask(self, teacher_feat, teacher_heatmap):
        """
        核心修复2：基于特征序列长度生成掩码
        args:
            teacher_feat: 教师特征 [bs, seq_len_feat, d_model]（用于获取序列长度）
            teacher_heatmap: 教师热力图 [bs, H, W]（用于筛选目标区域）
        returns:
            search_mask: 适配特征长度的目标掩码 [bs, seq_len_feat]
        """
        bs, seq_len_feat, _ = teacher_feat.shape
        bs_hm, H, W = teacher_heatmap.shape

        # 1. 从热力图提取目标区域掩码（原始尺寸 [bs, H, W]）
        target_thresh = torch.topk(teacher_heatmap.view(bs, -1), k=int(0.2*H*W), dim=-1)[0][:, -1].unsqueeze(-1).unsqueeze(-1)
        hm_mask = (teacher_heatmap >= target_thresh).float()  # [bs, H, W]

        # 2. 展平热力图掩码并适配特征序列长度
        hm_mask_flat = hm_mask.view(bs, -1)  # [bs, H*W]
        # 用adaptive_avg_pool1d将掩码长度调整为特征序列长度
        search_mask = F.adaptive_avg_pool1d(
            hm_mask_flat.unsqueeze(1),  # [bs, 1, H*W]
            output_size=seq_len_feat     # 适配特征序列长度 seq_len_feat
        ).squeeze(1)  # [bs, seq_len_feat]

        # 3. 二值化（避免中间值，确保掩码是0/1分布）
        search_mask = (search_mask > 0.5).float()
        return search_mask

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict, out_dict_s = self.forward_pass(data)
        
        # compute losses
        loss, status = self.compute_losses(out_dict, out_dict_s, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1
        assert len(data['template_event']) == 1
        assert len(data['search_event']) == 1
        # assert len(data['template_event_images']) == 1
        # assert len(data['search_event_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (bs, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:]) 
            template_list.append(template_img_i)  # (bs, 3, 128, 128)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (bs, 3, 256, 256)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:]) 
        
        # event_template_list = []
        # for i in range(self.settings.num_template):
        #     event_template_img_i = data['template_event_images'][i].view(-1,
        #                                                      *data['template_event_images'].shape[2:])  # (batch, 3, 128, 128)
        #     # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
        #     event_template_list.append(event_template_img_i)

        # event_search_img = data['search_event_images'][0].view(-1, *data['search_event_images'].shape[2:])  # (batch, 3, 320, 320)
        # # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        template_event = data['template_event'][0].view(-1, *data['template_event'].shape[2:])
        search_event = data['search_event'][0].view(-1, *data['search_event'].shape[2:])

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
        # if len(event_template_list) == 1:
        #     event_template_list = event_template_list[0]

        out_dict = self.net_teacher(
                            template=template_list,
                            search=search_img,
                            event_template=template_event,
                            event_search=search_event,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)
        

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

    def compute_losses(self, out_dict, out_dict_s, gt_dict, return_status=True):
        """核心：使用实际输出字段计算损失"""
        # ---------------------- 1. 提取实际字段 ----------------------
        # 热力图（WKD-L输入 + 掩码生成）
        teacher_heatmap = out_dict['score_map'].squeeze(1)  # [bs, H, W]（去除通道维）
        student_heatmap = out_dict_s['s_score_map'].squeeze(1)  # [bs, H, W]

        # 特征（WKD-F输入）
        teacher_feat = out_dict['teacher_feature']  # [bs, seq_len_t, d_model]
        student_feat = out_dict_s['student_feature']  # [bs, seq_len_s, d_model]

        # 预测框（原始任务损失）
        pred_boxes = out_dict_s['s_pred_boxes']
        gt_bbox = gt_dict['search_anno'][-1]

        # ---------------------- 2. 保留HDETrack原始任务损失 ----------------------
        # 生成GT热力图
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # [bs, 1, H, W]

        # 边界框损失（GIoU + L1）
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, pred_boxes.size(1), 1)).view(-1, 4).clamp(min=0.0, max=1.0)
        giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

        # 位置损失（Focal）
        location_loss = self.objective['focal'](out_dict_s['s_score_map'], gt_gaussian_maps)

        # ---------------------- 3. WKD蒸馏损失 ----------------------
        # （1）WKD-L损失（仅用热力图）
        wkd_l_loss = self.wkd_l(teacher_heatmap, student_heatmap)

        # （2）WKD-F损失（特征 + 目标掩码）
        search_mask = self.generate_search_mask(teacher_feat, teacher_heatmap)  # [bs, seq_len]
        wkd_f_loss = self.wkd_f(teacher_feat, student_feat, search_mask)

        # ---------------------- 4. 总损失加权求和 ----------------------
        total_loss = (
            self.loss_weight['giou'] * giou_loss +
            self.loss_weight['l1'] * l1_loss +
            self.loss_weight['focal'] * location_loss +
            wkd_l_loss +
            wkd_f_loss
        )

        # ---------------------- 5. 日志状态 ----------------------
        if return_status:
            status = {
                "Loss/total": total_loss.item(),
                "Loss/giou": giou_loss.item(),
                "Loss/l1": l1_loss.item(),
                "Loss/location": location_loss.item(),
                "Loss/wkd_l": wkd_l_loss.item(),
                "Loss/wkd_f": wkd_f_loss.item(),
                "IoU": iou.detach().mean().item()
            }
            return total_loss, status
        else:
            return total_loss
        


