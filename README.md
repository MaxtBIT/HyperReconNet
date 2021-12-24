# HyperReconNet
PyTorch codes for reproducing the paper: **Lizhi Wang, Tao Zhang, Ying Fu, and Hua Huang, HyperReconNet: Joint Coded Aperture Optimization and Image Reconstruction for Compressive Hyperspectral Imaging, TIP, 2019.**[[Link]](https://ieeexplore.ieee.org/document/8552450)

## Abstract
Coded aperture snapshot spectral imaging (CASSI) system encodes the 3D hyperspectral image (HSI) within a single 2D compressive image and then reconstructs the underlying HSI by employing an inverse optimization algorithm, which equips with the distinct advantage of snapshot but usually results in low reconstruction accuracy. To improve the accuracy, existing methods attempt to design either alternative coded apertures or advanced reconstruction methods, but cannot connect these two aspects via a unified framework, which limits the accuracy improvement. In this paper, we propose a convolution neural network-based end-to-end method to boost the accuracy by jointly optimizing the coded aperture and the reconstruction method. On the one hand, based on the nature of CASSI forward model, we design a repeated pattern for the coded aperture, whose entities are learned by acting as the network weights. On the other hand, we conduct the reconstruction through simultaneously exploiting intrinsic properties within HSI-the extensive correlations across the spatial and spectral dimensions. By leveraging the power of deep learning, the coded aperture design and the image reconstruction are connected and optimized via a unified framework. Experimental results show that our method outperforms the state-of-the-art methods under both comprehensive quantitative metrics and perceptive quality.

## Data
In the paper, two benchmarks are utilized for training and testing. Please check them in [Link1(ICVL)](http://icvl.cs.bgu.ac.il/hyperspectral/) and [Link2(Harvard)](http://vision.seas.harvard.edu/hyperspec/). To start your work, make HDF5 files of the same length and place them in the correct path. The file structure is as follows:<br/>
>--data/<br/>
>>--ICVL_train/<br/>
>>>--trainset_1.h5<br/>
>>>...<br/>
>>>--trainset_n.h5<br/>
>>>--train_files.txt<br/>
>>>--validset_1.h5<br/>
>>>...<br/>
>>>--validset_n.h5<br/>
>>>--valid_files.txt<br/>

>>--ICVL_test/<br/>
>>>--test1/<br/>
>>>...<br/>
>>>--testn/<br/>

Note that, every image for testing is saved as several 2D images according to different channels. In addition, only the central areas with 512 * 512 * 31 are compared in testing.

## Environment
Python 3.6.2<br/>
CUDA 10.0<br/>
Torch 1.7.0<br/>
OpenCV 4.5.4<br/>
h5py 3.1.0<br/>
TensorboardX 2.4<br/>

## Usage
1. Download this repository via git or download the [zip file](https://github.com/MaxtBIT/HyperReconNet/archive/refs/heads/main.zip) manually.
```
git clone https://github.com/MaxtBIT/HyperReconNet.git
```
2. Download the pre-trained models from [Model](https://drive.google.com/file/d/1wsyO5XKe6dD2RIm85u8afnJCpGmKVM8D/view?usp=sharing) if you need.
3. Make the datasets and place them in the correct path. Then, adapt the settings in **utils.py** according to your data.
4. Run the file **main.py** to train the model.
5. Run the file **test.py** to test the model.

## Citation
```
@article{HyperReconNet,<br/>
  title={HyperReconNet: Joint Coded Aperture Optimization and Image Reconstruction for Compressive Hyperspectral Imaging}, <br/>
  author={Wang, Lizhi and Zhang, Tao and Fu, Ying and Huang, Hua},<br/>
  journal={IEEE Transactions on Image Processing}, <br/>
  volume={28},<br/>
  number={5},<br/>
  pages={2257-2270},<br/>
  year={2019},<br/>
}
```
