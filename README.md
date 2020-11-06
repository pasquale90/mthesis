# Audio signal classification using deep learning algorithms

In this thesis we compare the performance of multiple feature parameters for environmental sound classification problems by developing multiple evaluating models. Specifically, as audio representation of 2 different datasets, we used waveforms, log-mel spectrograms and short-time Fourier transforms. Finally we set 4 different experiments and each one of them was divided in two discrete audio representation modes. For their evaluation and also for comparability purposes we developed hybrid CNN models. Along with comparing each mode within each experiment, we also compared the performances achieved by using each different dataset through inspecting and examining the factors of structure, the technical features and various prospects of the initial data distribution, respectively for each dataset. The nature of this research served us the additional opportunity to seek for potential environmental class-conditional audio features.

### Contents

- [Datasets](#datasets)
- [Audio_representations](#features)
- [Data_augmentation](#augmentation)
- [Method](#method)
- [Architecture](#architecture)
- [Models](#models)
- [Results](#results)

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

Audio data augmentation: White Noise, Time Stretching, Time Shifting, Pitch Shifting
Image data augmentation (2D exps) : Random Flip, Random Erasing


#### Method

General method scheme concerning all experiments:

<p align="center">
<img src="pics/method.png" width="700px"/>
</p>

#### Architecture

<p><br></p> 

<h3 align="center">1.raw architecture scheme</h3>
<p align="center">
<img src="pics/1.raw.png" width="700px"/>
</p>
 
<h3 align="center">2.flat architecture scheme</h3>
<p align="center">
<img src="pics/2.flat.png" width="700px"/>
</p>

<h3 align="center">3.mel architecture scheme</h3>
<p align="center">
<img src="pics/3.mel.png" width="700px"/>
 
<br></p>

<h3 align="center">4.stfts architecture scheme</h3>
<p align="center">
<img src="pics/4.stfts.png" width="700px"/>
 </p>




 
