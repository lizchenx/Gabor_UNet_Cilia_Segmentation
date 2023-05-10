# Gabor_UNet_Cilia_Segmentation
This is github repository for MSAI thesis work: Cilia Segmentation Using U-Net with Gabor Filter
## Abstract
Cilia are hairlike organelles found on the surface of most cell types. Cilia segmentation is fundamental to many biological studies and primary ciliary dyskinesia (PCD) diagnosis. Previously, biologists and clinicians detected and classified cilia manually, which was time consuming and error-prone. Previous studies have been using traditional and deep learning-based image segmentation methods for segmenting cilia. In this study, we propose to use Gabor filters (GFs) to perform feature extraction and train the U-Net model for cilia segmentation. We show that the composite models with the combination of Gabor filtered features improve the performances of the U-Net base model. Our best composite model with Gabor #25 ($\theta$ = $\pi$/2, $\sigma$=1, $\lambda$ = $\pi$*3/4, $\gamma$ = 0.05, $\phi$=0) achieved an IoU of 0.37, almost 8\% of improvement of the performance of the base model. By comparing the performances of the GF composite model with previous studies analyzing the cilia video set, we show that the presented framework outperformed previous models in terms of IoU score. Using GF as a data augmentation tool can help to enhance the robustness of features, and achieve a better performance. 

## Cilia data:
- Subset from previous work (Quinn et al. 2015)

## Python scripts for:
- Gabor Filter
- U-Net
- Evaluation: IoU, accuracy, F1, precision, recall

## References
- Quinn, Shannon P., et al. "Automated identification of abnormal respiratory ciliary motion in nasal biopsies." Science translational medicine 7.299 (2015): 299ra124-299ra124.
- Lu, Charles, et al. "Stacked Neural Networks for end-to-end ciliary motion analysis." arXiv preprint arXiv:1803.07534 (2018).
- Zain, Meekail, et al. "Towards an unsupervised spatiotemporal representation of cilia video using a modular generative pipeline." Proceedings of the Python in Science Conference. 2020.
- Zain, Meekail, et al. "Low Level Feature Extraction for Cilia Segmentation." Proceedings of the Python in Science Conference. 2022.
