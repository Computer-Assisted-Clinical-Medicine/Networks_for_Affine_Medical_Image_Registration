# Networks for Affine Medical Image Registration

This repository contains the implementation of 20 neural networks for affine registration of medical images, which were originally published in research papers. Since three-dimensional data was used, all networks were adapted accordingly. In order to provide a benchmark for comparison, we integrated our own neural network into the framework. The framework was implemented using Keras with Tensorflow backend.

If you use our code in your work please cite the following paper:
A. Strittmatter et al., "Deep Learning-Based Affine Medical Image Registration for Multimodal Minimal-Invasive Image-Guided Interventions - A Comparative Study on Generalizability", Z Med Phys 2023, https://doi.org/10.1016/j.zemedi.2023.05.003

# Benchmark Network

As benchmark, we used our own developed CNN, a ResNet-18 encoder and a U-Net-like decoder with one convolution per convolutional block without skip connections:

![BenchmarkCNNArchitecture](https://user-images.githubusercontent.com/129390849/230046201-3c61d409-0470-42b2-820c-8ec1ced254a1.png)


# Implemented Networks

Chee, E., Wu, Z., 2018. AIRNet: Self-Supervised Affine Registration for 3D Medical Images using Neural Networks. CoRR abs/1810.02583. URL: http://arxiv.org/abs/1810.02583, arXiv:1810.02583.

Chen, J., Frey, E.C., He, Y., Segars, W.P., Li, Y., Du, Y., 2022. TransMorph: Transformer for unsupervised medical image registration. Medical Image Analysis 82, 102615. doi:10.1016/j.media.2022.102615.

Chen, X., Meng, Y., Zhao, Y., Williams, R., Vallabhaneni, S.R., Zheng, Y., 2021b. Learning Unsupervised Parameter-Specific Affine Transformation for Medical Images Registration. pp. 24–34. doi:10.1007/978-3-030-87202-1\_3.

Gao, X., Van Houtte, J., Chen, Z., Zheng, G., 2021. DeepASDM: a Deep Learning Framework for Affine and Deformable Image Registration Incorporating a Statistical Deformation Model, in: 2021 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI), pp. 1–4. doi:10.1109/BHI50953.2021.9508553.

Gu, D., Liu, G., Tian, J., Zhan, Q., 2019. Two-Stage Unsupervised Learning Method for Affine and Deformable Medical Image Registration, in: 2019 IEEE International Conference on Image Processing (ICIP), pp. 1332–1336. doi:10.1109/ICIP.2019.8803794

Hasenstab, K.A., Cunha, G.M., Higaki, A., Ichikawa, S., Wang, K., Delgado, T., Brunsing, R.L., Schlein, A., Bittencourt, L.K., Schwartzman, A., Fowler, K.J., Hsiao, A., Sirlin, C.B., 2019. Fully automated convolutional neural network-based affine algorithm improves liver registration and lesion co-localization on hepatobiliary phase T1-weighted MR images. Eur Radiol Exp. 3(1):43. doi:10.1186/s41747-019-0120-7.

Hu, Y., Modat, M., Gibson, E., Li, W., Ghavami, N., Bonmati, E., Wang, E., Bandula, S., Moore, C.M., Emberton, M., Ourselin, S., Noble, J.A., Barratt, D.C., Vercauteren, T., 2018. Weakly-supervised convolutional neural networks for multimodal image registration. Medical Image Analysis 49, 1–13. doi:10.1016/j.media.2018.07.002.

Luo, G., Chen, X., Shi, F., Peng, Y., Xiang, D., Chen, Q., Xu, X., Zhu, W., Fan, Y., 2020. Multimodal affine registration for ICGA and MCSL fundus images of high myopia. Biomedical Optics Express 11. doi:10.1364/BOE. 393178.

Mok, T.C.W., Chung, A.C.S., 2022. Affine Medical Image Registration with Coarse-to-Fine Vision Transformer. 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , 20803–20812.

Roelofs, T.J.T., 2021. Deep Learning-Based Affine and Deformable 3D Medical Image Registration. Master’s thesis. Aalto University. Espoo, Finnland.

Shao, W., Banh, L., Kunder, C.A., Fan, R.E., Soerensen, S.J., Wang, J.B., Teslovich, N.C., Madhuripan, N., Jawahar, A., Ghanouni, P., Brooks, J.D., Sonn, G.A., Rusu, M., 2021. ProsRegNet: A deep learning framework for registration of MRI and histopathology images of the prostate. Medical Image Analysis 68, 101919. doi:10.1016/j.media.2020. 101919. 

Shen, Z., Han, X., Xu, Z., Niethammer, M., 2019. Networks for Joint Affine and Non-Parametric Image Registration, in: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE Computer Society, Los Alamitos, CA, USA. pp. 4219–4228. doi:10.1109/CVPR.2019. 00435.

de Silva, T., Chew, E.Y., Hotaling, N., Cukras, C.A., 2020. Deep-Learning based Multi-Modal Retinal Image Registration for Longitudinal Analysis of Patients with Age-related Macular Degeneration. Biomedical Optics Express 12. doi:10.1364/BOE.408573.

Tang, K., Li, Z., Tian, L., Wang, L., Zhu, Y., 2020. ADMIR–Affine and Deformable Medical Image Registration for Drug-Addicted Brain Images. IEEE Access 8, 70960–70968. doi:10.1109/ACCESS.2020.2986829.

Venkata, S.P., Duffy, B.A., Datta, K., 2022. An unsupervised deep learning method for affine registration of multi-contrast brain MR images. ISMRM 2022.

de Vos, B.D., Berendsen, F.F., Viergever, M.A., Sokooti, H., Staring, M., Isgum, I., 2019. A deep learning framework for unsupervised affine and deformable image registration. Medical Image Analysis 52, 128–143. doi:10.1016/j.media.2018.11.010. 

Waldkirch, B.I., 2020. Methods for three-dimensional Registration of Multimodal Abdominal Image Data. Ph.D. thesis. Ruprecht Karl University of Heidelberg.

Zeng, Q., Fu, Y., Tian, Z., Lei, Y., Zhang, Y., Wang, T., Mao, H., Liu, T., Curran, W., Jani, A., Patel, P., Yang, X., 2020. Label-driven MRI-US registration using weakly-supervised learning for MRI-guided prostate radiotherapy. Physics in Medicine & Biology 65. doi:10.1088/1361-6560/ab8cd6.

Zhao, S., Lau, T., Luo, J., Chang, E.I.C., Xu, Y., 2020. Unsupervised 3D End-to-End Medical Image Registration With Volume Tweening Network. IEEE Journal of Biomedical and Health Informatics 24, 1394–1404. doi:10.1109/jbhi.2019.2951024.

Zhu, Z., Cao, Y., Chenchen, Q., Rao, Y., Di, L., Dou, Q., Ni, D., Wang, Y.,2021. Joint affine and deformable three-dimensional networks for brain MRI registration. Medical Physics 48. doi:10.1002/mp.14674.

# Testdata
Test datasets are available here:

Zöllner, Frank, 2022, "Multimodal ground truth datasets for abdominal medical image registration [data]", https://doi.org/10.11588/data/ICSFUS, heiDATA, V1 
