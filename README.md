# Optimizing Gradient-driven Criteria in Network Sparsity: Gradient is All You Need ([Paper Link](https://arxiv.org/abs/2201.12826))

## Requirements

- Python >= 3.7.4
- Pytorch >= 1.6.1
- Torchvision >= 0.4.1

## Reproduce the Experiment Results

Select a configuration file in `configs` to reproduce the experiment results reported in the paper. For example, to prune ResNet-50 on ImageNet dataset, run:

   `python imagenet.py --config configs/resnet50_imagenet/90sparsity30epoch.yaml --gpus 0`

   Note that the `data_path` and `prune_rate` in the yaml file should be changed to the data path and your target sparse rate. 


Any problem, feel free to contact yuxinzhang@stu.xmu.edu.cn