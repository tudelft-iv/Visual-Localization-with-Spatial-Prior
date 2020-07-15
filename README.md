# Visual-Localization-with-Spatial-Prior

This repository contains codes for "Geographically Local Representation Learningwith a Spatial Prior for Visual Localization", ECCV MLAD workshop 2020

## Abstract
We revisit end-to-end representation learning for cross-viewself-localization, the task of retrieving for a query camera image the clos-est satellite image in a database by matching them in a shared imagerepresentation space. Previous work tackles this task as a global localiza-tion problem, i.e. assuming no prior knowledge on the location, thus thelearned image representation must distinguish far apart areas of the map.However, in many practical applications such as self-driving vehicles, itis already  possible to  discard distant locations through well-known  lo-calization techniques using temporal filters and GNSS/GPS sensors. Weargue that learned features should therefore be optimized to be discrimi-native within the geographic local neighborhood, instead of globally. We propose a simple but  effective  adaptation  to  the  common  triplet  lossused in previous work to consider a prior localization estimate alreadyin the training phase. We evaluate our approach on the existing CVACTdataset,  and  on  a  novel  localization  benchmark  based  on  the  OxfordRobotCar dataset which tests generalization across multiple drives anddays in the same area. For the Oxford benchmarks we collected corre-sponding  satellite  images.  With  a  localization  prior,  our  approach  im-proves recall@1 by 9 percent points on CVACT, and reduces the medianlocalization error by 2.45 meters on the Oxford benchmark, compared toa state-of-the-art baseline approach. Qualitative results underscore thatwith our approach the network indeed captures different aspects of thelocal surroundings compared to the global baseline.

## Datasets

## Models

## Codes
