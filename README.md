# Abstract
RGB cameras excel at capturing rich texture details and high spatial resolution, whereas event cameras offer exceptional temporal resolution and high dynamic range (HDR). Leveraging these complementary advantages can significantly improve object tracking under challenging scenarios, including high-speed motion, HDR environments, and dynamic background interference. However, substantial spatiotemporal asymmetry exists between these two modalities due to their fundamentally different imaging mechanisms, hindering effective multi-modal integration. To address this critical issue, we propose Hierarchical Asymmetric Distillation (HAD), a novel multi-modal knowledge distillation framework that explicitly models and conquers asymmetry information in spatio-temporal dimensions. Specifically, HAD introduces a hierarchical alignment strategy that mitigates information loss while maintaining the student network's computational efficiency and parameter count. Extensive experimental evaluations demonstrate the superiority of HAD over existing methods, and comprehensive ablation studies validate the effectiveness and necessity of each designed component.




# Environment 

Install env
```
conda create -n hadtrack python=3.8
conda activate hadtrack
bash install.sh
```

Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

Then, put the tracking datasets in `./data`. 

Download pre-trained MAE ViT-Base weights and teacher pre-trained in [EventVOT_Benchmark](https://github.com/Event-AHU/EventVOT_Benchmark) put it under `$/pretrained_models`

## Train & Test
```
# train
python tracking/train.py --script hadtrack --config hadtrack_eventvot --save_dir ./output --mode single --nproc_per_node 1 --use_wandb 0

# test
python tracking/test.py hdetrack hadtrack_eventvot --dataset eventvot --threads 1 --num_gpus 1
```


# Acknowledgement 
* Thanks for the [EventVOT](https://github.com/Event-AHU/EventVOT_Benchmark), [CEUTrack](https://github.com/Event-AHU/COESOT), [OSTrack](https://github.com/botaoye/OSTrack), [PyTracking](https://github.com/visionml/pytracking) and [ViT](https://github.com/rwightman/pytorch-image-models) library for a quickly implement.

# Citation 
```bibtex
@misc{deng2025hadhierarchicalasymmetricdistillation,
      title={HAD: Hierarchical Asymmetric Distillation to Bridge Spatio-Temporal Gaps in Event-Based Object Tracking}, 
      author={Yao Deng and Xian Zhong and Wenxuan Liu and Zhaofei Yu and Jingling Yuan and Tiejun Huang},
      year={2025},
      eprint={2510.19560},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.19560}, 
}
```
































