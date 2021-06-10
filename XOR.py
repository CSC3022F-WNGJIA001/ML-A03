# CSC3022F 2021 ML Assignment 3
# Part 1: XOR Problem
# Author: WNGJIA001

import random
from Perceptron import Perceptron

if __name__ == '__main__':
    num_gate = 0
    num_train = 100
    num_valid = 100

    # AND Gate
    AND = Perceptron(2, bias=-1.0)

    # generating training set
    training_examples = []
    training_labels = []
    for i in range(num_train):
        training_examples.append([random.random(), random.random()])
        # noise tolerance: all examples where x1 and x2 > 0.75 -> 1.0
        training_labels.append(1.0 if training_examples[i][0] > 0.75 and training_examples[i][1] > 0.75 else 0.0)

    # generating validation set
    validate_examples = []
    validate_labels = []
    for i in range(num_train):
        validate_examples.append([random.random(), random.random()])
        # noise tolerance: all examples where x1 and x2 > 0.75 -> 1.0
        validate_labels.append(1.0 if validate_examples[i][0] > 0.75 and validate_examples[i][1] > 0.75 else 0.0)

    # start training AND gate
    print("Training GATE_%1d..." % (num_gate))
    # print(AND.weights)
    valid_percentage = AND.validate(validate_examples, validate_labels, verbose=False)
    # print(valid_percentage)
    i = 0
    while valid_percentage < 0.98: # want AND Perceptron to have an accuracy of at least 98%
        i += 1
        AND.train(training_examples, training_labels, 0.2)  # Train our Perceptron
        # print('------ Iteration ' + str(i) + ' ------')
        # print(AND.weights)
        valid_percentage = AND.validate(validate_examples, validate_labels, verbose=False) # Validate it
        # print(valid_percentage)
        if i == 100: break
    print("\ttook ", i, " iterations to achieve a valid percentage of ", valid_percentage)

    # NOT Gate
    num_gate += 1
    NOT = Perceptron(1, bias=0.75)

    # generating training set
    training_examples = []
    training_labels = []
    for i in range(num_train):
        training_examples.append([random.random()])
        # noise tolerance: all examples where x1 > 0.75 -> 0.0
        training_labels.append(0.0 if training_examples[i][0] > 0.75 else 1.0)

    # generating validation set
    validate_examples = []
    validate_labels = []
    for i in range(num_train):
        validate_examples.append([random.random()])
        # noise tolerance: all examples where x1 > 0.75 -> 0.0
        validate_labels.append(0.0 if validate_examples[i][0] > 0.75 else 1.0)

    # start training NOT gate
    print("Training GATE_%1d..." % (num_gate))
    # print(NOT.weights)
    valid_percentage = NOT.validate(validate_examples, validate_labels, verbose=False)
    # print(valid_percentage)
    i = 0
    while valid_percentage < 0.98: # want NOT Perceptron to have an accuracy of at least 98%
        i += 1
        NOT.train(training_examples, training_labels, 0.2)  # Train our Perceptron
        # print('------ Iteration ' + str(i) + ' ------')
        # print(NOT.weights)
        valid_percentage = NOT.validate(validate_examples, validate_labels, verbose=False) # Validate it
        # print(valid_percentage)
        if i == 100: break
    print("\ttook ", i, " iterations to achieve a valid percentage of ", valid_percentage)

    # OR Gate
