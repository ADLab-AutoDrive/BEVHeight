<p align="center">

  <h1 align="center">BEVHeight: A Robust Framework for Vision-based Roadside 3D Object Detection</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=EUnI2nMAAAAJ&hl=zh-CN"><strong>Lei Yang</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=Jtmq_m0AAAAJ&hl=zh-CN&oi=sra"><strong>Kaicheng Yu</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=1ltylFwAAAAJ&hl=zh-CN&oi=sra"><strong>Tao Tang</strong></a>
    ·
    <a href="https://www.tsinghua.edu.cn/"><strong>Jun Li</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=aMnHLz4AAAAJ&hl=zh-CN&oi=ao"><strong>Kun Yuan</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=kLTnwAsAAAAJ&hl=zh-CN&oi=sra"><strong>Li Wang</strong></a>
    ·
    <a href="https://scholar.google.com.hk/citations?user=0Q7pN4cAAAAJ&hl=zh-CN&oi=sra"><strong>Xinyu Zhang</strong></a>
    ·
    <a href="https://damo.alibaba.com/labs/intelligent-transportation"><strong>Peng Chen</strong></a>
  </p>


<h2 align="center">CVPR 2023</h2>
  <div align="center">
    <img src="./assets/teaser_intro.png" alt="Logo" width="88%">
  </div>

<p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
     <a href='https://hub.docker.com/repository/docker/yanglei2024/op-bevheight/general'><img src='https://img.shields.io/badge/Docker-9cf.svg?logo=Docker' alt='Docker'></a>
    <br></br>
    <a href="https://arxiv.org/abs/2303.08498">
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
  </p>
</p>

**BEVHeight** is a new vision-based 3D object detector specially designed for roadside scenario. BEVHeight surpasses BEVDepth base-
line by a margin of 4.85% and 4.43% on **DAIR-V2X-I** and **Rope3D** benchmarks under the traditional clean settings, and by **26.88%** on robust settings where external camera parameters changes. We hope our work can shed light on studying more effective feature representation on **roadside perception**.


# News

- [2023/03/15] Both arXiv and codebase are released!
- [2023/02/27] BEVHeight got accepted to CVPR 2023!

# Incoming

- [ ] Release the pretrained models
- [ ] Support train and test on a custom dataset

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Getting Started">Getting Started</a>
    </li>
    <li>
      <a href="#Acknowledgment">Acknowledgment</a>
    </li>
    <li>
      <a href="#Citation">Citation</a>
    </li>
  </ol>
</details>

<br/>

# Getting Started

- [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_dataset.md)

Train BEVHeight with 8 GPUs
```
python [EXP_PATH] --amp_backend native -b 8 --gpus 8
```
Eval BEVHeight with 8 GPUs
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 8 --gpus 8
```

# Acknowledgment
This project is not possible without the following codebases.
* [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth)
* [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)
* [pypcd](https://github.com/dimatura/pypcd)

# Citation
If you use BEVHeight in your research, please cite our work by using the following BibTeX entry:
```
@inproceedings{yang2023bevheight,
    title={BEVHeight: A Robust Framework for Vision-based Roadside 3D Object Detection},
    author={Yang, Lei and Yu, Kaicheng and Tang, Tao and Li, Jun and Yuan, Kun and Wang, Li and Zhang, Xinyu and Chen, Peng},
    booktitle={IEEE/CVF Conf.~on Computer Vision and Pattern Recognition (CVPR)},
    month = mar,
    year={2023}
}
```
