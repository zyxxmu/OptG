# OptG: Optimizing Gradient-driven Criteria in Network Sparsity ([Paper Link](https://arxiv.org/abs/2201.12826))

## Requirements

- Python >= 3.7.4
- Pytorch >= 1.6.1
- Torchvision >= 0.4.1

## Reproduce the Experiment Results

Select a configuration file in `configs` to reproduce the experiment results reported in the paper. For example, to prune ResNet-50 on ImageNet dataset, run:

   `python imagenet.py --config configs/resnet50_imagenet/90sparsity30epoch.yaml --gpus 0`

   Note that the `data_path` and `prune_rate` in the yaml file should be changed to the data path and your target sparse rate. 

Evaluate Our Sparse Models

Our sparse models and training logs can be downloaded from the links in the following table. To test them, run:

`python imagenet.py --config configs/resnet50_imagenet/90sparsity30epoch.yaml --evaluate --evaluate_model_link <the sparse model link> --gpus 0`

| Model        | Sparsity | FLOPs | Top-1 Acc. | Link                                                         |
| ------------ | -------- | ----- | ---------- | ------------------------------------------------------------ |
| ResNet-50    | 90%      | 342M  | 74.55%     | [link](https://drive.google.com/drive/folders/1Yg0EjJNIMDXSFUsz76s3QSTQTYaOpq23?usp=share_link) |
| ResNet-50    | 95%      | 221M  | 72.45%     | [link](https://drive.google.com/drive/folders/1XbEorBPakpqpIfMFILD_kgCCluxZqJZf?usp=share_link) |
| ResNet-50    | 96.5%    | 179M  | 70.85%     | [link](https://drive.google.com/drive/folders/1Cmd8E6IPXxWOJh_EpaOAExLbeY1_UNGD?usp=sharing) |
| ResNet-50    | 98%      | 126M  | 67.20%     | [link](https://drive.google.com/drive/folders/11hGczoGg3db0VvUteNY42oiqLWBZwHUG?usp=sharing) |
| ResNet-50    | 99%      | 83M   | 62.10%     | [link](https://drive.google.com/drive/folders/1t-0zrR0pudTpqSN2rqH1sAbrdBPlpUdp?usp=sharing) |
| MobileNet-V1 | 80%      | 124M  | 70.27%     | [link](https://drive.google.com/drive/folders/1AdhfHTsr1SFdphhaGh-mjzIFo7ZTjnwU?usp=sharing) |
| MobileNet-V1 | 90%      | 80M   | 66.80%     | [link](https://drive.google.com/drive/folders/1DU9x2YAAFvTwajT76yjIqzb8QqyOSXYk?usp=sharing) |

Any problem, feel free to contact yuxinzhang@stu.xmu.edu.cn
