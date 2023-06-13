Semantic Segmentation of Urban Scene using UNET
==============================================================
==============================================================

This project is an implementation of the UNET architecture for image segmentation of the Cityscape dataset. The objective of this challenge is to develop a model that can accurately identify the diverse street scenes captured in various cities, with pixel-level annotations for different semantic categories, which can be useful in various applications such as self-driving cars, traffic monitoring, and parking lot management.


Model Architecture
------------------

The UNET architecture is a popular deep-learning architecture for image segmentation. It consists of an encoder and a decoder network, with a bottleneck layer in between. The encoder network downsamples the input image by applying convolutional and max pooling layers, while the decoder network upsamples the output of the bottleneck layer using transposed convolutional layers. Skip connections are also used to concatenate feature maps from the encoder network with those from the decoder network, allowing the model to retain high-level features while also preserving spatial information.

This implementation of the UNET architecture consists of four encoder blocks and four decoder blocks, with each block containing two convolutional layers followed by batch normalization and ReLU activation. The bottleneck layer is also a double convolutional layer with batch normalization and ReLU activation. The output layer is a single convolutional layer with sigmoid activation, which outputs a probability map for each pixel in the input image.

![UNET-architecture](u-net-architecture.png)

Dataset
-------

To use this project, you will need to download the Cityscapes dataset separately from the official website which can be found [here](https://www.cityscapes-dataset.com/).

Getting Started
---------------
Create a virtual environment in the root directory:

`pip install virtualenv`

`virtualenv <env name>`

To activate the environment:

`source <env name>/bin/activate`

Clone this repository:

`git clone https://github.com/Ashwij3/Pytorch_UNET_segmentation.git`

Next, install the required packages:

`pip install -r requirements.txt`

Training
--------

To train the model, run the following command:

Copy code

`python3 train.py`

By default, the training script will use the UNET architecture and the Cityscape dataset. You can customize the parameters of the training process by editing the `config.yaml` file.

Evaluation
----------

To evaluate the model on the test set, run the following command:

Copy code

`python3 evaluate.py`

This will generate a set of segmentation masks for the test images and calculate the accuracy of the model.


Acknowledgments
---------------

-   This project is based on the UNET architecture proposed by Ronneberger et al. in their paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" (<https://arxiv.org/abs/1505.04597>)
