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
    <img src="./assets/teaser_intro.jpg" alt="Logo" width="88%">
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
# Experimental Results


- DAIR-V2X-I Dataset
<div align=left>
<table>
     <tr align=center>
        <td rowspan="3">Method</td> 
        <td rowspan="3" align=center>Config File</td> 
        <td rowspan="3" align=center>Range</td> 
        <td colspan="3" align=center>Car</td>
        <td colspan="3" align=center>Pedestrain</td>
        <td colspan="3" align=center>Cyclist</td>
        <td rowspan="3" align=center>model pth</td>
    </tr>
    <tr align=center>
        <td colspan="3" align=center>3D@0.5</td>
        <td colspan="3" align=center>3D@0.25</td>
        <td colspan="3" align=center>3D@0.25</td>
    </tr>
    <tr align=center>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
    </tr>
    <tr align=center>
        <td rowspan="4">BEVHeight</td> 
         <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td>[0, 102.4]</td> 
        <td>77.48</td>
        <td>65.46</td>
        <td>65.53</td>
        <td>26.86</td>
        <td>25.53</td>
        <td>25.66</td>
        <td>51.18</td>
        <td>52.43</td>
        <td>53.07</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/6998b0b000aa45a0861e/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td><a href=exps/dair-v2x/bev_height_lss_r50_864_1536_128x128_140.py>R50_140</td>
        <td>[0, 140.8]</td> 
        <td>80.80</td>
        <td>75.23</td>
        <td>75.31</td>
        <td>28.13</td>
        <td>26.73</td>
        <td>26.88</td>
        <td>49.63</td>
        <td>52.27</td>
        <td>52.98</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/4fa0578a8c7347ebb353/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td><a href=exps/dair-v2x/bev_height_lss_r101_864_1536_256x256_102.py>R101_102</td>
        <td>[0, 102.4]</td> 
        <td>78.06</td>
        <td>65.94</td>
        <td>65.99</td>
        <td>40.45</td>
        <td>38.70</td>
        <td>38.82</td>
        <td>57.61</td>
        <td>59.90</td>
        <td>60.39</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/acd81d6083b742ddbb64/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td><a href=exps/dair-v2x/bev_height_lss_r101_864_1536_256x256_140.py>R101_140</td>
        <td>[0, 140.8]</td> 
        <td>81.80</td>
        <td>76.19</td>
        <td>76.26</td>
        <td>38.79</td>
        <td>37.94</td>
        <td>38.26</td>
        <td>58.22</td>
        <td>60.49</td>
        <td>61.03</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/9a0f179055724f5db6a3/?dl=1">model</a></td>
    </tr>
<table>
</div>

- Rope3D Dataset
<td><a href="https://cloud.tsinghua.edu.cn/f/f29279cebcbd4b3c8fb6/?dl=1">hom_train.pkl</a></td> <td><a href="https://cloud.tsinghua.edu.cn/f/3a9f3c7294794456b92d/?dl=1">hom_val.pkl</a></td>

<div align=center>
<table>
     <tr align=center>
        <td rowspan="2">Method</td> 
        <td rowspan="2" align=center>Config File</td> 
        <td rowspan="2" align=center>Range</td> 
        <td colspan="3" align=center>Car | 3D@0.5</td>
        <td colspan="3" align=center>Big Vehicle | 3D@0.5</td>
        <td colspan="3" align=center>Car | 3D@0.7</td>
        <td colspan="3" align=center>Big Vehicle | 3D@0.7</td>
        <td rowspan="2" align=center>model pth</td>
    </tr>
    <tr align=center>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
        <td>Easy</td>
        <td>Mod.</td>
        <td>Hard</td>
    </tr>
    <tr align=center>
        <td rowspan="4">BEVHeight</td> 
         <td><a href=exps/rope3d/bev_height_lss_r50_864_1536_128x128_102.py>R50_102</td>
        <td>[0, 102.4]</td> 
        <td>83.49</td>
        <td>72.46</td>
        <td>70.17</td>
        <td>50.73</td>
        <td>47.81</td>
        <td>47.80</td>
        <td>48.12</td>
        <td>42.45</td>
        <td>42.34</td>
        <td>24.58</td>
        <td>26.25</td>
        <td>26.28</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/fa3e2d07d62a44b7a337/?dl=1">model</a></td>
    </tr>
    <tr align=center>
        <td><a href=exps/rope3d/bev_height_lss_r50_864_1536_128x128_140.py>R50_140</td>
        <td>[0, 140.8]</td> 
        <td>85.46</td>
        <td>79.15</td>
        <td>79.06</td>
        <td>64.38</td>
        <td>65.75</td>
        <td>65.77</td>
        <td>46.39</td>
        <td>42.85</td>
        <td>42.71</td>
        <td>27.21</td>
        <td>33.99</td>
        <td>34.03</td>
        <td><a href="https://cloud.tsinghua.edu.cn/f/343be049d5e74d14a5af/?dl=1">model</a></td>
    </tr>
</table>
</div>

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
