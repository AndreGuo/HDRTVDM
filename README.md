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

## 0. Our scope

There're many HDR-related methods in this year's CVPR. Our method differs from others in that we take conventional SDR image to HDR in PQ/BT.2020 container (which is called *HDRTV* by [HDRTVNet(ICCV21)](http://openaccess.thecvf.com/content/ICCV2021/papers/Chen_A_New_Journey_From_SDRTV_to_HDRTV_ICCV_2021_paper.pdf)), and is meant to be applied in media industry.

Others methods may take single SDR to a linear-light-HDR in grapghics/rendering application, or merge several SDRs to single HDR which should be applied in camera imaging pipeline.
Please jump to them if you are interested in other HDR-related application scenario.

## 1. HDRTV4K Dataset

### 1.1 Training set

Our major concerns on training data are:

| Aspect                                                       |                                   Model's benefit                                    |
|:------------------------------------------------------------:|:------------------------------------------------------------------------------------:|
| (1) Label HDR's (scene) diversity                            |                            better generalization ability                             |
| (2) Label HDR's quality<br>(especially the amount of advanced color and luminance volume)|    more chance to produce advanced HDR/WCG volume        |
| (3) SDR's extent of degradation                              |                         a proper degradation recovery ability                        |
| (4) style and aesthetic of degraded SDR                      |                   better aesthetic performance<br>(or consistency from SDR)             |

Hence, we provide ***HDRTV4K*** label HDR (3848 individual frames) of better (1) quality and (2) diversity, you can downlaod from below:

[part1](https://userscloud.com/wbuaxcxaq19l), [part2](https://userscloud.com/eadf6gazjkm6), [part3](https://userscloud.com/a9j83lvpw10n), [part4](https://userscloud.com/8qtx3d4opav7), [part5](https://userscloud.com/1f08hwr4qj4g), [part6](https://userscloud.com/w1uhctigqwq9), [part7](https://userscloud.com/j0lb7vtinfks), [part8](https://userscloud.com/42nbjfv6fhfx), [part9](https://userscloud.com/anij44zvn4yx), [part10](https://userscloud.com/cxmtftz6rrl8), [part11](https://userscloud.com/yvi9t9jwc014), [part12](https://userscloud.com/nmhazmgkqbge) (UserClould, total 90.94GB, please extract these zip files together).

or

[here](https://pan.baidu.com/s/1YgskYN7-TFBlrh94WNbnyA?pwd=ny0p) (BaiduNetDisk, code:ny0p, please extract these zip files together)

Atfer obtaining label HDR, you can:

**1.1.1. Download the coresponding degraded SDR below:**

| From degradation model (DM) | (3) Extent of degradation | (4) Style or aesthetic | Download |
|:----:|:---------------------:|:---------------:|:--------:|
| ***OCIO2*** (ours)     | moderate                      | good                | [here(AliyunDrive)](https://www.aliyundrive.com/s/taNs6JGhAVj) or<br>[here(GoogleDrive)](https://drive.google.com/file/d/1eUCqMvUv-dBxHCQ1pW72sw4RvQDbI1o3/view?usp=sharing) (2.27GB)       |
| ***2446c+GM*** (ours)    | moderate                      | good                | [here(AliyunDrive)](https://www.aliyundrive.com/s/taNs6JGhAVj) or<br>[here(GoogleDrive)](https://drive.google.com/file/d/1UouhVb05NfMh8Z7gx3z1v_RcaoSmQEqA/view?usp=sharing) (2.03GB)       |
| ***HC+GM*** (ours)    | more                      | moderate                | [here(AliyunDrive)](https://www.aliyundrive.com/s/UXmUonpgukX) or<br>[here(GoogleDrive)](https://drive.google.com/file/d/1-qv7YPpM3sc_6Nex4UefWv6gEPBe2P2I/view?usp=sharing) (2.13GB)       |
| ***2446a*** (old)    | less                      | bad                |          |
| ***Reinhard*** (old)    | less                      | moderate                |          |
| ***YouTube*** (old, most widely adopted)    | more                      | bad                | [here(GoogleDrive)](https://drive.google.com/file/d/1_MuSt3mdpNlqcKp8so_qJMvbVCfGjlyG/view?usp=sharing) (2.51GB)<br>(if used, you can learn a silimar style as previous methods)        |
| ***DaVinci*** (w. different settings) | less | good | this DM is not discussed in our paper, TODO |

and use any of them to train your network (since AliyunDrive donnot support sharing .zip, file there will be a .exe self-extract package and you can run it at Window system).

Since our degradation models (DMs) are just a preliminary attempt on concerns (3) and (4), we encourage you to:

**1.1.2. (Encouraged) Use your own degradation model to obtain input SDR**

In this case, you can:

+ Change the style and aesthetic of degraded SDR to better suit your own technical and artistic intention, or involve your expertise in color science *etc.* for more precise control between SDR and HDR. 
+ Control the extent of degradation to follow the staticstics of target SDR in your own application scenario (*e.g.* remastering legacy SDR or converting on-the-air SDR). You can even add diversity on the extent of degradation to endow your network a generalizability to various extent of degradation.
+ Add new types of degradation *e.g.* camera noise, compression artifact, motion blur, chromatic aberration and film grain *etc.* for more specific application scenario. Their degradation models are relatively studied more and you can easily find more references.

### 1.2 Test set

+ The test set used in our paper (consecutive frames) is protected by copyright and will not be relesed. In this case, we provided alternative test set which consists of 400 individual frames from 10% training set [here(AliyunDrive)](https://www.aliyundrive.com/s/QodPeQyJ3C2) or [here(GoogleDrive)](https://drive.google.com/file/d/15VbRZeKVztG4Q_ovVvzo0LzGmW0ThjJf/view?usp=sharing).
+ In our paper, conventional distance-based metrics *PSNR*, *SSIM*, *deltaE* and *VDP* don't work since SDR-HDR numerical relation in training and test set is different (This is like model trained on *ExpertC* of *Adobe-MIT-5K* dataset will score lower on *ExpertA*). So if you want these metrics work, you should test on the same test set (*i.e.* if your model is trained with ***OCIO2*** SDR, you should also test it on ***OCIO2*** SDR).
+ From the prespective of quality assessment (QA), the assessment of ITM/up-conversion (enhancement process) is still an open task. We and our colleague is currently working on a better benchmark, and will update here if it's released.

## 2. Method

### 2.0 Note that

- The name of our neural network is *LSN (luminance segmented network)*
- Curently *LSN* is only trained on our own data, if you want to compare it on current benchmark e.g. HDRTV1K, please wait us releasing the checkpoint trained on current training set e.g. HDRTV1K. 

### 2.1 Prerequisites

- Python
- PyTorch
- OpenCV
- ImageIO
- NumPy

### 2.2 Usage (how to test)

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

## 3. Assessment criteria of HDR/WCG container and ITM process

In our paper we use 4 HDR/WCG exclusive metrics to measure how many HDR/WCG volume a single frame possess.

| Dimension                                                    |                                   Spatial fraction                                   |                         Numerical energy                         |
|:------------------------------------------------------------:|:------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
| HDR (high dynamic range) volume                              |   [FHLP](/metrics/HDRdegreeAssessment.m)(Fraction of HighLight Pixels)               |    [EHL](/metrics/HDRdegreeAssessment.m)(Extent of HighLight)    |
| WCG (wide color gamut) volume                                |   [FWGP](/metrics/WCGdegreeAssessment.m)(Fraction of Wide Gamut Pixels)              |    [EWG](/metrics/WCGdegreeAssessment.m)(Extent of Wide Gamut)   |

You can find their usage in the comment.

**Note that**: 
From the prespective of quality assessment (QA), these metrics have not been proven to be consistently positively-correlated with good viewing experience, therefore the should only serve as a reference of HDR/WCG volume.
HDR/WCG's preception involoves sophisticated knowlegde in color science and human vision *etc.*, and intuitively these 4 metrics chould be mesured in a "naturalness" way (counting FHLP/EHL/FWGP/EWG's distribution on large-scale visually-pleasuring HDR/WCG images, and juding if someone's FHLP/EHL/FWGP/EWG falls in commom distribution.)

TO BE UPDATED

## Contact

Guo Cheng ([Andre Guo](https://orcid.org/orcid=0000-0002-2660-2267)) guocheng@cuc.edu.cn

- *State Key Laboratory of Media Convergence and Communication (MCC),
Communication University of China (CUC), Beijing, China.*
- *Peng Cheng Laboratory (PCL), Shenzhen, China.*
