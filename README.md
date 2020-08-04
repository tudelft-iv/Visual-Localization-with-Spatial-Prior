# Visual-Localization-with-Spatial-Prior

This repository contains codes for "Geographically Local Representation Learningwith a Spatial Prior for Visual Localization", ECCV Map-based Localization for Autonomous Driving Workshop 2020

## Abstract
We revisit end-to-end representation learning for cross-view self-localization, the task of retrieving for a query camera image the closest satellite image in a database by matching them in a shared image representation space. Previous work tackles this task as a global localization problem, i.e. assuming no prior knowledge on the location, thus the learned image representation must distinguish far apart areas of the map. However, in many practical applications such as self-driving vehicles, it is already possible to discard distant locations through well-known localization techniques using temporal filters and GNSS/GPS sensors. We
argue that learned features should therefore be optimized to be discriminative within the geographic local neighborhood, instead of globally. We propose a simple but effective adaptation to the common triplet loss used in previous work to consider a prior localization estimate already in the training phase. We evaluate our approach on the existing CVACT dataset, and on a novel localization benchmark based on the Oxford RobotCar dataset which tests generalization across multiple traversals and days in the same area. For the Oxford benchmarks we collected corresponding satellite images. With a localization prior, our approach improves recall@1 by 9 percent points on CVACT, and reduces the median localization error by 2.45 meters on the Oxford benchmark, compared to a state-of-the-art baseline approach. Qualitative results underscore that with our approach the network indeed captures different aspects of the local surroundings compared to the global baseline.

## Datasets
The CVACT dataset can be accessed from: https://github.com/Liumouliu/OriCNN

## Models
Models trained on CVACT can be find through the link: https://drive.google.com/drive/folders/1F520jxdU6zQIxGk0dM4ygDVGIWvNzq8Y?usp=sharing

If you want to train the model by yourself, you can initialize the graph with the model provides in the folder "Initialize". It is the same initialization in the released code of the baseline method: https://github.com/shiyujiao/cross_view_localization_SAFA, where the VGG part is pre-trained on the Imagenet, and other parts are initialized randomly.

## Codes
