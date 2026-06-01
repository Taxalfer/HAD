from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.coesot_path = '/root/shared-nvme/data/datasets/COESOT'
    settings.eventvot_path = '/root/shared-nvme/data/datasets/EventVOT'
    settings.fe108_path = '/root/shared-nvme/data/datasets/FE108'
    settings.network_path = '/root/shared-nvme/code/EventVOT_Benchmark/HDETrack/output/test/networks'    # Where tracking networks are stored.
    settings.prj_dir = '/root/shared-nvme/code/EventVOT_Benchmark/HDETrack'
    settings.result_plot_path = '/root/shared-nvme/code/EventVOT_Benchmark/HDETrack/output/test/result_plots'
    settings.results_path = '/root/shared-nvme/code/EventVOT_Benchmark/HDETrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/root/shared-nvme/code/EventVOT_Benchmark/HDETrack/output'
    settings.segmentation_path = '/root/shared-nvme/code/EventVOT_Benchmark/HDETrack/output/test/segmentation_results'
    settings.visevent_path = '/root/shared-nvme/data/datasets/VisEvent'

    return settings

