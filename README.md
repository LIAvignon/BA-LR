# BA-LR: Toward an interpretable and explainable approach for automatic voice comparison

## How to install
To install BA-LR, do the following:

0. Use a conda environment
1. Install requirements:
```sh
pip install -r requirement.txt
```
2. Clone repository:
```sh
git clone https://github.com/Imenbaa/BA-LR.git
```
## 1) BA-vectors extractor
The extractor is trained on Voxceleb2 dataset https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html. It is composed of a ResNet generator of speech representations optimised for speaker classification task. 
After training phase, we obtain sparse representations (0,x), we replace x to 1 to obtain binary representation.
#### Generator
`Filterbanks -> ResNet generator -> embedding -> Softplus layer() -> Sparse representation`  
#### Speaker Classifier
`Sparse representation -> classifier (i.e. NN projected to num_classes with Softmax) -> class prediction`
#### BA-Vector
`Sparse representation -> BA-vectors`

```sh
git clone https://github.com/Imenbaa/BA-LR.git
```
## 2) LR Framework

### BA bevavioral parameters



