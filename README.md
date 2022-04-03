# imagedenoising

ABSTRACT <br />
Image restoration is a highly active area of research, as smartphone cameras often provide noisy low quality images. Overall, Convolutional Neural Networks have proven to perform well as an image denoising technique, but skip-connection architectures perform particularly well [1], [2]. Hence, this
paper set out to identify the best performing architecture on an image restoration task. Furthermore, this paper investigated the effects of the skip-connection architecture on modelconvergence and performance. Finally, several loss functions were implemented to investigate how different loss functions result in different image restoration qualities. This paper found that the U-Net architecture with the Structural Similarity Index Measure loss function performed the best on the current task of image restoration.


INTRODUCTION <br />
Within the field of image restoration, convolutional neural networks (CNNs) have proved to perform well [1], [2]. In particular, architectures implementing skip-connections have performed well due to a focus on residual learning [1]. For that reason, this paper experimented with several different CNN architectures of increasing complexity with a focus on skip-connection architectures. Additionally, since the aim of this paper is to restore high quality images to facilitate the diagnosis process of human specialists, a big part of this pa- per was to identify a loss function addressing how the Human Visual System (HVS) is thought to work [3], [4].


RESULTS <br />
From Fig. 1 below it is evident that at the current noise level, specified in 2.1, only the small U-Net and ResNet-18 are powerful enough to getting close to recovering the clean image X from the degraded input image Y. The CNN and the two CNNs with a fully connected layer are nowhere near powerful enough to reconstruct X to a satisfactory degree. Hence, the required model performance is heavily dependent on the specified noise level. For that reason, this paper experimented with several different noise levels. On one hand the noise level had to be high enough to force the model to learn more robust features, but on the other hand the noise level also had to be representative of the noise level in the application images.

![Figure 1](https://github.com/MadsBirch/imagedenoising/blob/main/figures/best_model5.png?raw=true) <br />
*Figure 1*


CHOOSING APPROPRIATE LOSS FUNCTIONS <br />
Choosing an appropriate loss function is essential, because it defines the desired outcome of the model. In this section, it will be investigated how different loss functions result in different outcomes in terms of image quality. The figure visualizes image reconstructions of the small U-Net using a range of different loss functions; SSIM, MSE/L2, L1 and BCE. As discussed in 2.3, the MSE loss does not correlate well with how the HVS perceives image quality [5]. This lack of correspondence is due to the simplicity of the MSE loss. Contrary to the MSE loss, the SSIM loss is designed to better correlate with how HVS functions by taking luminance, contrast and structure into account [4].

From Fig. 2 it is evident that the SSIM loss results higher image quality due to a focus on reconstructing structural information. By focusing on structural information the resulting reconstruction is sharper with easily distinguishable edges. In contrast, the MSE loss resulted in blurry reconstructions with edges being less distinguishable (Fig. 7). Additionally, this paper found that the L1 loss slightly outperformed the MSE loss. The image quality of L1 is slightly higher in terms of color restoration and sharpness (Fig. 7). This finding is sim

![Figure 1](https://github.com/MadsBirch/imagedenoising/blob/main/figures/loss_results.png?raw=true) <br />
*Figure 2*

REFERENCES:<br />

[1] Syed Waqas Zamir, Aditya Arora, Salman Khan, Mu- nawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang, “Restormer: Efficient transformer for high- resolution image restoration,” 2021. <br />
[2] Linwei Fan, Fan Zhang, Hui Fan, and Caiming Zhang, “Brief review of image denoising techniques,” Visual Computing for Industry, Biomedicine, and Art, vol. 2, no. 1, pp. 1–12, 2019. <br />
[3] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli, “Image quality assessment: from error visi- bility to structural similarity,” IEEE transactions on im- age processing, vol. 13, no. 4, pp. 600–612, 2004. <br />
[4] Hang Zhao, Orazio Gallo, Iuri Frosio, and Jan Kautz, “Loss functions for image restoration with neural net- works,” IEEE Transactions on computational imaging, vol. 3, no. 1, pp. 47–57, 2016.
