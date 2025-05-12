<p align="center">

  <h1 align="center">FreeSplat: Generalizable 3D Gaussian Splatting Towards Free-View Synthesis of Indoor Scenes Reconstruction</h1>
  <p align="center">
    <a href="https://wangys16.github.io/">Yunsong Wang</a>,
    <a href="https://tianxinhuang.github.io/">Tianxin Huang</a>,
    <a href="https://hlinchen.github.io/">Hanlin Chen</a>,
    <a href="https://www.comp.nus.edu.sg/~leegh/">Gim Hee Lee</a>

  </p>

  <h2 align="center">NeurIPS 2024</h2>

  <h3 align="center"><a href="https://arxiv.org/pdf/2405.17958">arXiv</a> | <a href="https://wangys16.github.io/FreeSplat-project/">Project Page</a>  </h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./teaser/teaser.png" alt="Logo" width="95%">
  </a>
</p>

# 🌏 News
[01 April 2025] We have introduced an enhanced version, ***FreeSplat++***, to specifically focus on feed-forward whole scene reconstruction with 3D Gaussians. You may want to take a look at its [Project page](https://wangys16.github.io/FreeSplatPP-Page/), [Arxiv](https://arxiv.org/pdf/2503.22986) and [Code](https://github.com/wangys16/FreeSplatPP).

[01 April 2025] We updated the code for training evaluation on RE10K dataset.

[12 May 2025] We fixed the erros in training when using batch size > 1.

<br>

# 📃 Abstract

FreeSplat is a generalizable 3DGS method for indoor scene reconstruction, which leverages low-cost 2D backbones for feature extraction and cost volume for multi-view aggregation. Furthermore, FreeSplat proposes a Pixel-wise Triplet Fusion (PTF) module to merge multi-view 3D Gaussians, such that to remove those redundant ones and provide point-level latent fusion and regularization on Gaussian localization. FreeSplat shows consistent quality and efficiency improvements especially when given large numbers of input views.

<br>

# 📌 Installation

To get started, create a virtual environment using Python 3.10+:

```bash
git clone https://github.com/wangys16/FreeSplat.git
cd FreeSplat
conda create -n freesplat python=3.10
conda activate freesplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If your system does not use CUDA 12.1 by default, see the troubleshooting tips below from [pixelSplat](https://github.com/dcharatan/pixelsplat).

<details>
<summary>Troubleshooting</summary>
<br>

The Gaussian splatting CUDA code (`diff-gaussian-rasterization`) must be compiled using the same version of CUDA that PyTorch was compiled with. As of December 2023, the version of PyTorch you get when doing `pip install torch` was built using CUDA 12.1. If your system does not use CUDA 12.1 by default, you can try the following:

- Install a version of PyTorch that was built using your CUDA version. For example, to get PyTorch with CUDA 11.8, use the following command (more details [here](https://pytorch.org/get-started/locally/)):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- Install CUDA Toolkit 12.1 on your system. One approach (*try this at your own risk!*) is to install a second CUDA Toolkit version using the `runfile (local)` option [here](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local). When you run the installer, disable the options that install GPU drivers and update the default CUDA symlinks. If you do this, you can point your system to CUDA 12.1 during installation as follows:

```bash
LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64 pip install -r requirements.txt
# If everything else was installed but you're missing diff-gaussian-rasterization, do:
LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64 pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```
</details>

<br>

# 📁 Acquiring Datasets

FreeSplat is trained using about 100 scenes from [ScanNet](http://www.scan-net.org) following [NeRFusion](https://github.com/jetd1/NeRFusion) and [SurfelNeRF](https://github.com/TencentARC/SurfelNeRF), and evaluated on ScanNet and [Replica](https://github.com/facebookresearch/Replica-Dataset) datasets.

You can download our preprocessed ScanNet and Replica [here](https://drive.google.com/drive/folders/1_KqJnSfNrNxSMguBwFtR1cxTPxdLG7Sc?usp=sharing). For RE10K and ACID downloading, you should follow [pixelSplat](https://github.com/dcharatan/pixelsplat). The downloaded datasets under path ```datasets/``` should look like:
```
datasets
├─ scannet
│  ├─ train
│  ├  ├─scene0005_00
|  ├  ├  ├─ color (RGB images)
│  ├  ├  ├─ depth (depth images)
│  ├  ├  ├─ intrinsic (intrinsics)
│  ├  ├  └─ extrinsics.npy (camera extrinsics)
│  ├  ├─ scene0020_00
│  ├  ...
│  ├─ test
│  ├  ├─
│  ├  ...
│  ├─ train_idx.txt (training scenes list)
│  └─ test_idx.txt (testing scenes list)
├─ replica
│  ├─ test
│  └─ test_idx.txt (testing scenes list)
├─ re10k
│  ├─ train
│  ├─ test
├─ acid
│  ├─ train
│  ├─ test
```

Our sampled views for evaluation on different settings are in ```assets/evaluation_index_{dataset}_{N}views.json```.

# Acquiring Pre-trained Checkpoints

You can find our pre-trained checkpoints [here](https://drive.google.com/drive/folders/1NKmXXeyTkTeiAsnOcwmWV-1dxuBdyBTb?usp=sharing) and download them to path ```checkpoints/```.


# 🎮 Running the Code

## Training
### ScanNet

The main entry point is `src/main.py`. To train FreeSplat on 2-views, 3-views, and FVT settings, you can respectively call:

```bash
python -m src.main +experiment=scannet/2views +output_dir=train_2views
```
```bash
python -m src.main +experiment=scannet/3views +output_dir=train_3views
```
```bash
python -m src.main +experiment=scannet/fvt +output_dir=train_fvt
```
The output will be saved in path ```outputs/***```.

### RE10K
To train on RE10K, you can use: 
```bash
python -m src.main +experiment=re10k/2views +output_dir=train_re10k_2views
``` 
```bash
python -m src.main +experiment=re10k/fvt +output_dir=train_re10k_fvt
```
to train on 2-views or FVT setting.

## Evaluation
### ScanNet and Replica

To evaluate pre-trained model on the ```[N]```-views setting on ```[DATASET]```, you can call:

```bash
python -m src.main +experiment=[DATASET]/[SETTING] +output_dir=[OUTPUT_PATH] mode=test dataset/view_sampler=evaluation checkpointing.load=[PATH_TO_CHECKPOINT] dataset.view_sampler.num_context_views=[N]
```

For example, to evaluate 2-views trained FreeSplat:

```bash
python -m src.main +experiment=scannet/2views +output_dir=test_scannet_2views mode=test dataset/view_sampler=evaluation checkpointing.load=checkpoints/2views.ckpt dataset.view_sampler.num_context_views=2
```
To evaluate FreeSplat-fvt on ScanNet 10-views setting, you can run:

```bash
python -m src.main +experiment=scannet/fvt +output_dir=test_scannet_fvt mode=test dataset/view_sampler=evaluation checkpointing.load=checkpoints/fvt.ckpt dataset.view_sampler.num_context_views=10 model.encoder.num_views=9
```

Here ```model.encoder.num_views=9``` is to use more nearby views for more accurate depth estimation. We also provide a whole scene reconstruction example that you can possibly run by:
```bash
python -m src.main +experiment=scannet/fvt +output_dir=test_scannet_whole mode=test dataset/view_sampler=evaluation checkpointing.load=checkpoints/fvt.ckpt dataset.view_sampler.num_context_views=30 model.encoder.num_views=30
```

### RE10K

To evaluate 2-views results of your trained ```re10k_2views.ckpt``` checkpoint on RE10K, you can run:
```bash
python -m src.main +experiment=re10k/2views +output_dir=test_re10k_2views dataset/view_sampler=evaluation checkpointing.load=checkpoints/re10k_2views.ckpt dataset.view_sampler.num_context_views=2
```
<br>

# Camera Ready Updates

1. Our current version directly uses the features extracted by the backbone to conduct multi-view matching, achieving faster training and better performance with slight GPU overhead. 
2. For Gaussian Splatting codebase, we now follow [diff-gaussian-rasterization-w-depth](https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth) for more accurate depth rendering.


<br>


# 🖊 BibTeX
If you find our work helpful, please consider citing our papers. Thank you!

```
@article{wang2025freesplat++,
  title={FreeSplat++: Generalizable 3D Gaussian Splatting for Efficient Indoor Scene Reconstruction},
  author={Wang, Yunsong and Huang, Tianxin and Chen, Hanlin and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2503.22986},
  year={2025}
}
```

```
@article{wang2024freesplat,
  title={FreeSplat: Generalizable 3D Gaussian Splatting Towards Free-View Synthesis of Indoor Scenes},
  author={Wang, Yunsong and Huang, Tianxin and Chen, Hanlin and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2405.17958},
  year={2024}
}
```

# 📜 Acknowledgements

Our code is largely based on [pixelSplat](https://github.com/dcharatan/pixelsplat), and our implementation also referred to [SimpleRecon](https://github.com/nianticlabs/simplerecon) and [MVSplat](https://github.com/donydchen/mvsplat). Thanks for their great works!
