# Multi-granularity Temporal Modeling for Partially Relevant Video Retrieval


## Catalogue 
* [1. Getting Started](#getting-started)
* [2. Run](#run)
* [3. Results](#results)


## Getting Started

1. Clone this repository:

```
git clone https://github.com/tianxinlin666/MTM.git
cd MTM
```

2. Create a conda environment and install the dependencies:

> Since the installation of the Mamba library often encounters network issues, I have provided **a series of pre-downloaded local installation packages** for you.
>
> 
>
> For users within China, you may consider using Baidu Netdisk links for downloads.
>
> > 通过网盘分享的文件：package
> > 链接: https://pan.baidu.com/s/1NGueqLXQAtAIpETzlZgrbw?pwd=3a7a 提取码: 3a7a
>
> 
>
> You can also choose to download directly through the link We provided. (Note: This link is slower than Baidu Netdisk.)
>
> > http://120.26.160.25/package/

```shell
conda create -n mamfusion-env python=3.10
conda activate mamfusion-env
pip install torch-2.4.1+cu118-cp310-cp310-linux_x86_64.whl
pip install torchaudio-2.4.1+cu118-cp310-cp310-linux_x86_64.whl
pip install torchvision-0.19.1+cu118-cp310-cp310-linux_x86_64.whl
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
pip install mamba_ssm-2.2.1+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.3.0.post1+cu118torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
```

3. Download Datasets: All features of TVR, ActivityNet Captions and Charades-STA are kindly provided by the authors of [MS-SL].

4. Set dataset location

## Run

To train MTM on TVR:
```shell
cd src
python main.py -d tvr
```

To train MTM on ActivityNet Captions:
```shell
cd src
python main.py -d act
```

To train MTM on Charades-STA:
```shell
cd src
python main.py -d cha
```

## Results

### Quantitative Results

For this repository, the expected performance is:

| *Dataset* | *R@1* | *R@5* | *R@10* | *R@100* | *SumR* |
| ---- | ---- | ---- | ---- | ---- | ---- |
| TVR | 16.0 | 38.4 | 49.2 | 86.8 | 190.4 |
| ActivityNet Captions | 9.1 | 27.3 | 40.4 | 79.3 | 156.1 |
| Charades-STA | 2.6 | 9.5 | 14.8 | 54.2 | 81.1 |



We have also provided you with pre-trained models for verification.

> http://120.26.160.25/Pre-trained-model/



[MS-SL]:https://github.com/HuiGuanLab/ms-sl

