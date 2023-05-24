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

## 1. HDRTV4K Dataset

### 1.1 Training set

Our major concerns on training data are:

| Aspect                                                       |                                   Model's benefit                                    |
|:------------------------------------------------------------:|:------------------------------------------------------------------------------------:|
| (1) Label HDR's (scene) diversity                            |                            better generalization ability                             |
| (2) Label HDR's quality<br>(especially the amount of advanced color and luminance volume)|    more chance to produce advanced HDR/WCG volume        |
| (3) SDR's extent of degradation                              |                         a proper degradation recovery ability                        |
| (4) style and aesthetic of degraded SDR                      |                   better aesthetic performance (or consistency from SDR)             |

Hence, we provide ***HDRTV4K*** label HDR of better diversity and quality [here(AliyunDrive)](TODO).

Atfer obtaining label HDR, you can:

**1. Download the coresponding degraded SDR below:**

| From degradation model (DM) | Extent of degradation | Style/aesthetic | Download |
|:----:|:---------------------:|:---------------:|:--------:|
| ***OCIO2*** (ours)     | moderate                      | good                | [here(AliyunDrive)](https://www.aliyundrive.com/s/taNs6JGhAVj) or [here(UserClould)](https://userscloud.com/5twsnxw6zgoz) (2.27GB)       |
| ***2446c+GM*** (ours)    | moderate                      | good                | [here(AliyunDrive)](https://www.aliyundrive.com/s/taNs6JGhAVj) or [here(UserClould)](https://userscloud.com/fdy4ohopf11k) (2.03GB)       |
| ***HC+GM*** (ours)    | good                      | moderate                | [here(AliyunDrive)](https://www.aliyundrive.com/s/UXmUonpgukX) or [here(UserClould)](https://userscloud.com/r0j891m6lqpc) (2.13GB)       |
| ***2446a*** (old)    | bad                      | bad                |          |
| ***Reinhard*** (old)    | bad                      | moderate                |          |
| ***YouTube*** (old, most widely adopted)    | good                      | bad                |          |

and use any of them to train your network. Since our degradation models (DMs) are just a preliminary attempt on concerns (3) and (4), we encourage you to:

**2. (Encouraged) Use your own degradation model to obtain input SDR**

In this case, you can:

+ Change the style and aesthetic of degraded SDR to better suit your own intention, or involve your expertise in color science *etc.* for more precise control between SDR and HDR. 
+ Control the extent of degradation to follow the staticstics of target SDR in your own application scenario (*e.g.* remastering legacy SDR or converting on-the-air SDR). You can even add diversity on the extent of degradation to endow your network a generalizability to various extent of degradation.
+ Add new types of degradation *e.g.* camera noise, compression artifact, motion blur and film grain *etc.* for more specific application scenario. Their degradation modles are relatively studied more and you can easily find more references.

### 1.2 Test set

+ The test set used in our paper (consecutive frames) is protected by copyright and will not be relesed. In this case, we provided alternative test set which consists of 400 individual frames from 10% training set [here(AliyunDrive)](https://www.aliyundrive.com/s/QodPeQyJ3C2) or [here(UserClould)](https://userscloud.com/n6t3da6mtfzy).
+ In our paper, conventional distance-based metrics *PSNR*, *SSIM*, *deltaE* and *VDP* don't work since SDR-HDR numerical relation in training and test set is different (This is like model trained on *ExpertC* of *Adobe-MIT-5K* dataset will score lower on *ExpertA*). So if you want these metrics work, you should test on the same test set (*i.e.* if your model is trained with ****OCIO2*** SDR, you should also test it on ***OCIO2*** SDR).
+ From the prespective of quality assessment (QA), the assessment of ITM/up-conversion is still an open task. We and our colleague is currently working on a better benchmark, and will update here if it's released.

## 2. Method

TO BE UPDATED

## 3. Assessment criteria of HDR/WCG container and ITM process

TO BE UPDATED

## Contact

Guo Cheng ([Andre Guo](https://orcid.org/orcid=0000-0002-2660-2267)) guocheng@cuc.edu.cn

- *State Key Laboratory of Media Convergence and Communication (MCC),
Communication University of China (CUC), Beijing, China.*
- *Peng Cheng Laboratory (PCL), Shenzhen, China.*
