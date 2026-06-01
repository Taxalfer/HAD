class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/root/shared-nvme/code/EventVOT_Benchmark/HDETrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/root/shared-nvme/code/EventVOT_Benchmark/HDETrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/root/shared-nvme/code/EventVOT_Benchmark/HDETrack/pretrained_networks'
        self.coesot_dir = '/root/shared-nvme/data/datasets/COESOT/train'
        self.coesot_val_dir = '/root/shared-nvme/data/datasets/COESOT/test'
        self.fe108_dir = '/root/shared-nvme/data/datasets/FE108/train'
        self.fe108_val_dir = '/root/shared-nvme/data/datasets/FE108/test'
        self.visevent_dir = '/root/shared-nvme/data/datasets/VisEvent/train_subset'
        self.visevent_val_dir = '/root/shared-nvme/data/datasets/VisEvent/test_subset'
        self.eventvot_dir = '/root/shared-nvme/data/datasets/EventVOT/EventVOT_train'
