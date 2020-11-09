# Audio signal classification using deep learning algorithms
<div align="center">

[![University](https://img.shields.io/badge/University-%CE%91%CE%A0%CE%98-red.svg)](http://ee.auth.gr/)
[![Subject](https://img.shields.io/badge/Subject-environmental%20audio%20classification-lightgrey.svg)](https://github.com/pasquale90/mthesis)
[![Subject](https://img.shields.io/badge/computational%20resources-%CE%91%CF%81%CE%B9%CF%83%CF%84%CE%BF%CF%84%CE%AD%CE%BB%CE%B7%CF%82-brightgreen.svg)](https://hpc.it.auth.gr/)
[![City](https://img.shields.io/badge/City-Thessaloniki-blue.svg)](https://github.com/pasquale90/mthesis)
[![Year](https://img.shields.io/badge/Year-2020-yellow.svg)](https://github.com/pasquale90/mthesis)



</div>

In this thesis we compared the performance of multiple feature parameters for environmental sound classification problems by developing multiple evaluating models. Specifically, as audio representation of 2 different datasets, we used waveforms, log-mel spectrograms and short-time Fourier transforms. Finally we set 4 different experiments and each one of them was divided in two discrete audio representation modes. For their evaluation and also for comparability purposes we developed hybrid CNN models. Along with comparing each mode within each experiment, we also compared the performances achieved by using each different dataset through inspecting and examining the factors of structure, the technical features and various prospects of the initial data distribution, respectively for each dataset. The nature of this research additionally enabled us to seek for potential environmental class-conditional audio features.

### Contents

- [Datasets](#datasets)
- [Audio_representations](#features)
- [Data_augmentation](#augmentation)
- [Method](#method)
- [Models](#models)
- [Results](#results)

#### Folders' Structure

<p align="left">
<img src="contents.jpg" width="700px"/>
</p>
         


#### Tools

- code implementation : [python3] (https://www.python.org/) 
- environment/packages : [miniconda3] (https://docs.conda.io/projects/conda/en/latest/)
- framework : [pytorch] (https://github.com/pytorch/pytorch)
- feature extraction/synthetic data : [librosa] (https://librosa.org/)
- image augmentation : [torchvision](https://github.com/pytorch/vision)

#### Datasets

- ESC-50 : (https://github.com/karolpiczak/ESC-50)
- UrbanSound8k : (https://urbansounddataset.weebly.com/urbansound8k.html)

#### Audio_representations

- 1.raw : 1D raw waveform
- 2.flat : 1D flattened log mel-spectogram
- 3.mel : 2D log mel-spectogram
- 4.stfts : 2D short-time Fourier transform

#### Data_augmentation

- Audio data augmentation: White Noise, Time Stretching, Time Shifting, Pitch Shifting
- Image data augmentation (2D exps) : Random Flip, Random Erasing


#### Method

<h4 align="center">General method scheme concerning all experiments:</h4>

<p align="center">
<img src="arch/method.png" width="700px"/>
</p>

#### Models

<h4 align="center">1.raw architecture scheme</h4>
<p align="center">
<img src="arch/1.raw.png" width="700px"/>
</p>
 
<h4 align="center">2.flat architecture scheme</h4>
<p align="center">
<img src="arch/2.flat.png" width="700px"/>
</p>

<h4 align="center">3.mel architecture scheme</h4>
<p align="center">
<img src="arch/3.mel.png" width="700px"/>


<h4 align="center">4.stfts architecture scheme</h4>
<p align="center">
<img src="arch/4.stfts.png" width="700px"/>
 </p>
 
 #### Results
 
<h4 align="center">Average</h4>
<p align="center">
<img src="Results/general.png" width="700px"/>
 </p>
 
<h4 align="center">
 <a href="https://github.com/pasquale90/mthesis/tree/master/Results">analytical fold results</a>
</h4>
 
<h4 align="center">
 <a href = "https://github.com/pasquale90/mthesis/tree/master/Results/Analytical">analytical class results</a>
</h4>
  




 
