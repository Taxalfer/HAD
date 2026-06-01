import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

class DualModalAnalyzer:
    def __init__(self, sequence_path, output_path):
        self.sequence_path = sequence_path
        self.output_path = output_path
        self.rgb_dir = os.path.join(sequence_path, f"{os.path.basename(sequence_path)}_aps")
        self.event_dir = os.path.join(sequence_path, f"{os.path.basename(sequence_path)}_dvs")
        self.results = []

    def load_frames(self):
        """加载RGB和事件帧（事件帧为RGB图像）"""
        rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith('.bmp') or f.endswith('.png')])
        event_files = sorted([f for f in os.listdir(self.event_dir) if f.endswith('.bmp') or f.endswith('.png')])
        
        print(f"找到 {len(rgb_files)} 个RGB帧和 {len(event_files)} 个事件帧")
        assert len(rgb_files) == len(event_files), "RGB和事件帧数量不匹配"
        
        frames = []
        for rgb_file, event_file in zip(rgb_files, event_files):
            # 以彩色模式读取事件帧
            rgb = cv2.imread(os.path.join(self.rgb_dir, rgb_file))
            event = cv2.imread(os.path.join(self.event_dir, event_file))
            
            # 确保RGB图像为3通道
            if rgb.shape[2] != 3:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            if event.shape[2] != 3:
                event = cv2.cvtColor(event, cv2.COLOR_BGR2RGB)
                
            frames.append((rgb, event))
        return frames

    def compute_edge_metrics(self, rgb, event):
        """计算边缘特征指标（事件帧为RGB）"""
        # 将事件帧转换为灰度图像进行边缘检测
        event_gray = cv2.cvtColor(event, cv2.COLOR_RGB2GRAY)
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 自适应Canny阈值
        v_rgb = np.median(rgb_gray)
        low_rgb, high_rgb = int(max(0, (1 - 0.33) * v_rgb)), int(min(255, (1 + 0.33) * v_rgb))
        rgb_edges = cv2.Canny(rgb_gray, low_rgb, high_rgb)

        v_event = np.median(event_gray)
        low_event, high_event = int(max(0, (1 - 0.33) * v_event)), int(min(255, (1 + 0.33) * v_event))
        event_edges = cv2.Canny(event_gray, low_event, high_event)

        # 动态掩码（事件帧非零即为运动区域）
        motion_mask = (event_gray > 0).astype(np.uint8)
        dynamic_rgb_edges = cv2.bitwise_and(rgb_edges, rgb_edges, mask=motion_mask)

        intersection = np.logical_and(event_edges, dynamic_rgb_edges).sum()
        union = np.logical_or(event_edges, dynamic_rgb_edges).sum()
        edge_iou = intersection / (union + 1e-6)
        return edge_iou

    def compute_texture_metrics(self, rgb, event):
        """计算纹理特征指标（事件帧为RGB）"""
        # 将事件帧转换为灰度
        event_gray = cv2.cvtColor(event, cv2.COLOR_RGB2GRAY)
        rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # 多方向/距离GLCM
        glcm_rgb = graycomatrix(rgb_gray, distances=[1, 2], angles=[0, np.pi/4], symmetric=True, normed=True)
        glcm_event = graycomatrix(event_gray, distances=[1, 2], angles=[0, np.pi/4], symmetric=True, normed=True)
        contrast_rgb = graycoprops(glcm_rgb, 'contrast')[0, 0]
        contrast_event = graycoprops(glcm_event, 'contrast')[0, 0]

        # 局部熵差异（使用RGB图像）
        entropy_rgb = shannon_entropy(rgb.reshape(-1, 3)).mean()
        entropy_event = shannon_entropy(event.reshape(-1, 3)).mean()
        entropy_diff = abs(entropy_rgb - entropy_event)

        return contrast_rgb, contrast_event, entropy_diff

    def compute_flow_metrics(self, prev_rgb, curr_rgb, prev_event, curr_event):
        """计算光流指标（事件帧为RGB）"""
        # 将RGB图像转换为灰度
        prev_rgb_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
        curr_rgb_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
        prev_event_gray = cv2.cvtColor(prev_event, cv2.COLOR_RGB2GRAY)
        curr_event_gray = cv2.cvtColor(curr_event, cv2.COLOR_RGB2GRAY)

        flow_rgb = cv2.calcOpticalFlowFarneback(prev_rgb_gray, curr_rgb_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_event = cv2.calcOpticalFlowFarneback(prev_event_gray, curr_event_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        epe = np.sqrt(np.sum((flow_rgb - flow_event)**2, axis=2)).mean()
        return epe

    def analyze_sequence(self):
        """分析整个视频序列"""
        frames = self.load_frames()
        for i in tqdm(range(1, len(frames)), desc="Processing frames"):
            prev_rgb, prev_event = frames[i-1]
            curr_rgb, curr_event = frames[i]

            edge_iou = self.compute_edge_metrics(curr_rgb, curr_event)
            contrast_rgb, contrast_event, entropy_diff = self.compute_texture_metrics(curr_rgb, curr_event)
            epe = self.compute_flow_metrics(prev_rgb, curr_rgb, prev_event, curr_event)

            self.results.append({
                'frame': i,
                'edge_iou': edge_iou,
                'contrast_rgb': contrast_rgb,
                'contrast_event': contrast_event,
                'entropy_diff': entropy_diff,
                'flow_epe': epe
            })

        df = pd.DataFrame(self.results)
        output_csv = os.path.join(self.output_path, "modal_analysis.csv")
        df.to_csv(output_csv, index=False)
        print(f"结果已保存至: {output_csv}")
        self.plot_results(df)

    def plot_results(self, df):
        """绘制指标变化曲线（生成四个独立子图）"""
        # 设置全局字体为 Times New Roman
        plt.rcParams['font.sans-serif'] = ['Times New Roman']  
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示问题
        plt.rcParams['font.family'] = 'serif'  # 确保使用衬线字体

        # 子图配置列表（包含每个子图的文件名）
        subplot_config = [
            {
                'filename': 'edge_iou_analysis.png',
                'ylabel': 'IoU',
                'title': 'Dynamic Edge IoU - Edge Consistency Comparison',
                'color': 'blue',
                'linestyle': '-',  # 实线
                'data_col': 'edge_iou'
            },
            {
                'filename': 'contrast_comparison.png',
                'ylabel': 'Contrast',
                'title': 'RGB vs Event Contrast - Texture Analysis',
                'colors': ['red', 'green'],
                'linestyles': ['-', '-'],  # 全部实线
                'data_cols': ['contrast_rgb', 'contrast_event']
            },
            {
                'filename': 'entropy_difference.png',
                'ylabel': 'Difference',
                'title': 'Entropy Difference - Information Comparison',
                'color': 'purple',
                'linestyle': '-',
                'data_col': 'entropy_diff'
            },
            {
                'filename': 'flow_epe_accuracy.png',
                'ylabel': 'EPE (px)',
                'title': 'Flow EPE - Pixel-Level Accuracy',
                'color': 'orange',
                'linestyle': '-',
                'data_col': 'flow_epe'
            }
        ]

        # 逐个生成并保存独立子图
        for config in subplot_config:
            # 创建新的画布（每个子图单独的画布）
            fig, ax = plt.subplots(figsize=(10, 6))  # 单图尺寸更适配
            
            # 设置刻度字体大小
            ax.tick_params(axis='both', labelsize=20)
            ax.grid(True, linestyle='--', alpha=0.5)  # 网格线保留虚线不影响
            
            # 设置标题和轴标签
            ax.set_title(config['title'], fontsize=18, fontweight='bold', fontfamily='Times New Roman')
            ax.set_xlabel('Frame Number', fontsize=16, fontweight='bold', fontfamily='Times New Roman')
            ax.set_ylabel(config['ylabel'], fontsize=16, fontweight='bold', fontfamily='Times New Roman')

            # 单指标绘制（删除了平均值计算和绘制）
            if 'data_col' in config:
                y_data = df[config['data_col']]
                # 只绘制核心曲线，移除平均值相关代码
                ax.plot(df['frame'], y_data, 
                        color=config['color'], linestyle=config['linestyle'])

            # 双指标绘制（删除了平均值计算和绘制）
            elif 'data_cols' in config:
                for i, data_col in enumerate(config['data_cols']):
                    y_data = df[data_col]
                    # 只绘制核心曲线，移除平均值相关代码
                    ax.plot(df['frame'], y_data,
                            color=config['colors'][i], linestyle=config['linestyles'][i])

            # 自动调整布局
            plt.tight_layout()
            
            # 保存单个子图
            plot_path = os.path.join(self.output_path, config['filename'])
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭画布释放资源
            print(f"子图已保存至: {plot_path}")

if __name__ == "__main__":
    dataset_path = '/root/shared-nvme/data/datasets/COESOT/train'
    sequence_name = 'dvSave-2021_12_21_17_34_12'
    sequence_path = os.path.join(dataset_path, sequence_name)
    output_path = os.path.join('/root/shared-nvme/code/EventVOT_Benchmark/HDETrack/visualization/analysis', sequence_name)
    # 判断路径是否存在，不存在则创建（exist_ok=True 避免路径已存在时报错）
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"创建输出目录: {output_path}")
    else:
        print(f"输出目录已存在: {output_path}")

    analyzer = DualModalAnalyzer(sequence_path, output_path)
    analyzer.analyze_sequence()