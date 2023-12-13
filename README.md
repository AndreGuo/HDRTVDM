# HDRTVDM
The official repo of paper
"***Learning a Practical SDR-to-HDRTV Up-conversion using New Dataset and Degradation Models***"
([paper (ArXiv)](https://arxiv.org/abs/2303.13031),
[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Guo_Learning_a_Practical_SDR-to-HDRTV_Up-Conversion_Using_New_Dataset_and_Degradation_CVPR_2023_paper.pdf),
[supplementary material](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Guo_Learning_a_Practical_CVPR_2023_supplemental.pdf))
in CVPR2023.
    
    @InProceedings{Guo_2023_CVPR,
        author    = {Guo, Cheng and Fan, Leidong and Xue, Ziyu and Jiang, Xiuhua},
        title     = {Learning a Practical SDR-to-HDRTV Up-Conversion Using New Dataset and Degradation Models},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023},
        pages     = {22231-22241}
    }

---

# 0. Our scope

There're many HDR-related methods in this year's CVPR. Our method differs from others in that we take conventional SDR image to HDR in PQ/BT.2020 container (which is called *HDRTV* by [HDRTVNet(ICCV21)](http://openaccess.thecvf.com/content/ICCV2021/papers/Chen_A_New_Journey_From_SDRTV_to_HDRTV_ICCV_2021_paper.pdf)), and is meant to be applied in media industry.

Others methods may take single SDR to a linear-light-HDR in grapghics/rendering application, or merge several SDRs to single HDR which should be applied in camera imaging pipeline.
Please jump to them if you are interested in other HDR-related application scenario.

# 1. HDRTV4K Dataset (Training set & test set)

## 1.1 Training set

Our major concerns on training data are:

| Aspect                                                       |                                   Model's benefit                                    |
|:------------------------------------------------------------:|:------------------------------------------------------------------------------------:|
| (1) Label HDR's (scene) diversity                            |                            better generalization ability                             |
| (2) Label HDR's quality<br>(especially the amount of advanced color and luminance volume)|    more chance to produce advanced HDR/WCG volume        |
| (3) SDR's extent of degradation                              |                         a proper degradation recovery ability                        |
| (4) style and aesthetic of degraded SDR                      |                   better aesthetic performance<br>(or consistency from SDR)             |

Hence, we provide ***HDRTV4K*** label HDR (3848 individual frames) of better (1) quality and (2) diversity, available on:

[BaiduNetDisk](https://pan.baidu.com/s/1YgskYN7-TFBlrh94WNbnyA?pwd=ny0p), GoogleDrive(TODO).

Atfer obtaining label HDR, you can:

### 1.1.1. Download the coresponding degraded SDR below:

| From degradation model (DM) | (3) Extent of degradation | (4) Style or aesthetic | Download |
|:----:|:---------------------:|:---------------:|:--------:|
| ***OCIO2*** (ours)     | moderate                      | good                | [GoogleDrive](https://drive.google.com/file/d/1eUCqMvUv-dBxHCQ1pW72sw4RvQDbI1o3/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1TAcILuuwn0PS8AVQTC3UjQ?pwd=fuu2) (2.27GB)       |
| ***2446c+GM*** (ours)    | moderate                      | good                | [GoogleDrive](https://drive.google.com/file/d/1UouhVb05NfMh8Z7gx3z1v_RcaoSmQEqA/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1uP9FmXWODun6LdUUu1jmGg?pwd=671z) (2.03GB)       |
| ***HC+GM*** (ours)    | more                      | moderate                | [GoogleDrive](https://drive.google.com/file/d/1-qv7YPpM3sc_6Nex4UefWv6gEPBe2P2I/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1zm27I0idMWML5F2YXO6EDQ?pwd=c9zg) (2.13GB)       |
| ***2446a*** (old)    | less                      | bad                |  [BaiduNetDisk](https://pan.baidu.com/s/1yY2L7S6cKeJ26P2Rn2tIFg?pwd=7vp7)        |
| ***Reinhard*** (old)    | less                      | moderate                | [BaiduNetDisk](https://pan.baidu.com/s/1JBdlPBLV8wZ6YXpOTbJC4g?pwd=w6p5)         |
| ***YouTube*** (old, most widely adopted)    | more                      | bad                | [GoogleDrive](https://drive.google.com/file/d/1_MuSt3mdpNlqcKp8so_qJMvbVCfGjlyG/view?usp=sharing), [BaiduNetDisk](https://pan.baidu.com/s/1tlMibrUCBVLoC7KmzzvfMg?pwd=s4dv) (2.51GB)<br>(if used, you can learn a silimar style as previous methods)        |
| ***DaVinci*** (w. different settings) | less | good | [GoogleDrive](https://drive.google.com/file/d/1DcHseQuQ9NZHs5qoWq8h77Q2CzfACjrN/view?usp=drive_link), [BaiduNetDisk](https://pan.baidu.com/s/1uKWpvprlOXWlnIHSrh9pyQ?pwd=v9kx)<br>(this DM is used in our another paper [ITM-LUT](https://github.com/AndreGuo/ITMLUT)) |

and use any of them to train your network (since AliyunDrive donnot support sharing .zip, file there will be a .exe self-extract package and you can run it at Window system).

Since our degradation models (DMs) are just a preliminary attempt on concerns (3) and (4), we encourage you to:

### 1.1.2. (Encouraged) Use your own degradation model to obtain input SDR

In this case, you can:

+ Change the style and aesthetic of degraded SDR to better suit your own technical and artistic intention, or involve your expertise in color science *etc.* for more precise control between SDR and HDR. 
+ Control the extent of degradation to follow the staticstics of target SDR in your own application scenario (*e.g.* remastering legacy SDR or converting on-the-air SDR). You can even add diversity on the extent of degradation to endow your network a generalizability to various extent of degradation.
+ Add new types of degradation *e.g.* camera noise, compression artifact, motion blur, [chromatic aberration](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Universal_and_Flexible_Optical_Aberration_Correction_Using_Deep-Prior_Based_Deconvolution_ICCV_2021_paper.pdf) and [film grain](https://arxiv.org/pdf/2206.07411v1.pdf) *etc.* for more specific application scenario. Their degradation models are relatively studied more with traditional and deep-learning model.

## 1.2 Test set

+ The test set used in our paper (consecutive frames) is copyrighted and will not be relesed. We provided alternative test set which consists of ***400 individual frames*** and even more scenes, it's available on:

[BaiduNetDisk](https://pan.baidu.com/s/19bs6KHfnOrT_t-hcMJXJQw?pwd=7917) and GoogleDrive(TODO)

+ In our paper, conventional distance-based metrics *PSNR*, *SSIM*, *deltaE* and *VDP* don't work since SDR-HDR/WCG numerical relation in training and test set is different (This is like model trained on *ExpertC* of *Adobe-MIT-5K* dataset will score lower on *ExpertA*). So if you want these metrics work, you should test on the same test set (*i.e.* if your model is trained with ***OCIO2*** SDR, you should also test it on ***OCIO2*** SDR).
+ You can only take our GT HDR and use your own degradation model to generate input SDR, to test different aspect of method performance.
+ From the prespective of quality assessment (QA), the assessment of ITM/up-conversion (enhancement process) is still an open task. We and our colleague is currently working on it, please refer to [here](https://www.sciencedirect.com/science/article/abs/pii/S0141938223001439) or [here](https://www.researchgate.net/publication/373316933_Inverse-tone-mapped_HDR_video_quality_assessment_A_new_dataset_and_benchmark).

# 2. Method

## 2.0 Note that

- The name of our neural network is **LSN (luminance segmented network)**
- Curently **LSN** is only trained on our own data, if you want to compare it on current benchmark e.g. HDRTV1K, please wait us releasing the checkpoint trained on current training set e.g. HDRTV1K. 

## 2.1 Prerequisites

- Python
- PyTorch
- OpenCV
- ImageIO
- NumPy

## 2.2 Usage (how to test)

Run `method/test.py` with below configuration(s):

```bash
python3 method/test.py frameName.jpg
```

When batch processing, use wildcard `*`:

```bash
python3 method/test.py framesPath/*.png
```

or like:

```bash
python3 method/test.py framesPath/footageName_*.png
```

Add below configuration(s) for specific propose:

| Purpose                                                                                          |                                    Configuration                                     |
|:-------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------:|
| Specifing output path                                                                            |                       `-out resultDir/` (default is inputDir)                        |
| Resizing image before inference                                                                  |                       `-resize True -height newH -width newW`                        |
| Adding filename tag                                                                              |                                    `-tag yourTag`                                    |
| Forcing CPU processing                                                                           |                                   `-use_gpu False`                                   |
| Using input SDR with bit depth != 8                                                              |                               *e.g.* `-in_bitdepth 16`                               
| Saving result HDR in other format<br/>(defalut is uncompressed<br/>16-bit `.tif`of single frame) | `-out_format suffix`<br>`png` as 16bit .png<br>`exr` require extra package `openEXR` |

We also provide alternative checkpoints: `method/params_YouTube.pth` (trained with same lable HDR, but different ***YouTube*** degradation model) and `method/params_Zeng20.pth` (trained with same degradation model, but different label HDR from ***Zeng20*** dadatset), to show the imapct of training set on our task.

If you want to refer to our network only, you can call it ***LSN*** (Luminance Segmented Network).

TO BE UPDATED

# 3. Assessment criteria of HDR/WCG container and ITM process

In our paper we use 4 metrics to measure how many HDR/WCG volume a single frame possess.

| Dimension                                                    |                                   Spatial fraction                                   |                         Numerical energy                         |
|:------------------------------------------------------------:|:------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
| HDR (high dynamic range) volume                              |   [FHLP](/metrics/HDRdegreeAssessment.m)(Fraction of HighLight Pixels)               |    [EHL](/metrics/HDRdegreeAssessment.m)(Extent of HighLight)    |
| WCG (wide color gamut) volume                                |   [FWGP](/metrics/WCGdegreeAssessment.m)(Fraction of Wide Gamut Pixels)              |    [EWG](/metrics/WCGdegreeAssessment.m)(Extent of Wide Gamut)   |

You can find their usage in the comment.

**Note that**: 
From the prespective of quality assessment (QA), these metrics have not been proven to be consistently positively-correlated with good viewing experience, therefore the should only serve as a reference of HDR/WCG volume.
HDR/WCG's preception involoves sophisticated knowlegde in color science and human vision *etc.*, and intuitively these 4 metrics chould be mesured in a "naturalness" way (counting FHLP/EHL/FWGP/EWG's distribution on large-scale visually-pleasuring HDR/WCG images, and juding if someone's FHLP/EHL/FWGP/EWG falls in commom distribution.)

TO BE UPDATED

# Contact

Guo Cheng ([Andre Guo](https://orcid.org/orcid=0000-0002-2660-2267)) guocheng@cuc.edu.cn

- *State Key Laboratory of Media Convergence and Communication (MCC),
Communication University of China (CUC), Beijing, China.*
- *Peng Cheng Laboratory (PCL), Shenzhen, China.*
