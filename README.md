# CSC3022F Machine Learning Assignment 3: Articial Neural Networks

The requirements.txt provides a list of packages that are required for setting
up the virtual environment.

A makefile has been provided to automate the building of the virtual env and the
removal the compiled files.

```ssh
make
```
creates the virtual environment and installs the necessary package in the
virtual environment for this project.

```ssh
make clean
```
removes the virtual environment and compilation files.

```ssh
source venv/bin/activate
```
activates the virtual environment

## Part 1: The XOR Problem

XOR.py implements the XOR gate but cascading 1 OR, 1 NOT and 2 AND gates.
the input values were assumed to be a random value from the ranges (-0.25~0.25)
and (0.75~1.25).

To run the script, use the following command:

```ssh
python3 XOR.py
```

## Part 2: Image Classication

Classifier.py implements an ANN that trains on the MNIST10 dataset to classify
a handwritten digit. The MNIST dataset should be placed with a folder named 'data'
and this folder should be in the same directory as the Classifier.py script.
More details regarding the design of the neural network can be found in the report.

To run the script, use the following command:

```ssh
python3 Classifier.py
```
