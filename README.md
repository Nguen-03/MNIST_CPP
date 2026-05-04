# MNIST_CPP

Neural network using C++ to classify digits from the MNIST dataset

## :bar_chart: Results

<br>
Dataset: MNIST dataset of torchvision (28x28 pixel images)
<p align="center">
    <img src="images/image.png" height=300 width=300>
    <img src="images/image1.png" height=400 width=400>
</p>

## :gear: Installation and Usage

- git clone https://github.com/Nguen-03/MNIST_CPP.git
- The number of dataset images could be configured in **mnist.py**
- Then type these commands in the terminal:
    - **mingw32-make** to compile the project
    - **./train.exe**
    - **./predict.exe**: print images and show result (the number of predicted images could be changed in **predict.cpp**)

## Requirements

- C++17
- MINGW / GCC(of course☝️🤓)

## How it works

1.  Input image (28x28) → flatten to vector (784)
2.  Forward pass through hidden layers
3.  (6)Output layer with Softmax
4.  Loss: Cross Entropy
5.  Backpropagation to update weights
6.  Save weight to **lmao.bin**

# Reference

- https://github.com/Magicalbat/videos/tree/main/machine-learning
