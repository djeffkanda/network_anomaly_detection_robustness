# Deep Learning for Network Anomaly Detection: Robustness Evaluation under Data Contamination

# Deep unsupervised anomaly detection algorithms
This repository collects different unsupervised machine learning algorithms to detect anomalies.
## Implemented models
We have implemented the following models. Our implementations of ALAD, DeepSVDD, 
DROCC and MemAE closely follows the original implementations already available on GitHub.
- [x] [AutoEncoder]()
- [x] [ALAD](https://arxiv.org/abs/1812.02288)
- [x] [DAGMM](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf)
- [x] [DeepSVDD](http://proceedings.mlr.press/v80/ruff18a.html)
- [x] [DSEBM](https://arxiv.org/abs/1605.07717)
- [x] [DROCC](https://arxiv.org/abs/2002.12718)
- [x] [DUAD](https://openaccess.thecvf.com/content/WACV2021/papers/Li_Deep_Unsupervised_Anomaly_Detection_WACV_2021_paper.pdf)
- [x] [LOF](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)
- [x] [MemAE](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_Memorizing_Normality_to_Detect_Anomaly_Memory-Augmented_Deep_Autoencoder_for_Unsupervised_ICCV_2019_paper.pdf)
- [x] [NeuTraLAD](https://arxiv.org/pdf/2103.16440.pdf)
- [x] [OC-SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)
- [x] [RecForest](https://epubs.siam.org/doi/pdf/10.1137/1.9781611976700.15)
- [x] [SOM-DAGMM](https://arxiv.org/pdf/2008.12686.pdf)

## Dependencies
A complete dependency list is available in requirements.txt.
We list here the most important ones:
- torch@1.10.2 with CUDA 11.3
- numpy
- pandas
- scikit-learn

## Installation
Assumes latest version of Anaconda was installed.
```
$ conda create --name [ENV_NAME] python=3.8
$ conda activate [ENV_NAME]
$ pip install -r requirements.txt
```
Replace `[ENV_NAME]` with the name of your environment.

## Usage
From the root of the project.
```
$ python -m src.main 
-m [model_name]
-d [/path/to/dataset/file.{npz,mat}]
--dataset [dataset_name]
--batch-size [batch_size]
```
