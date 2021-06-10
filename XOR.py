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
    print("Training GATE_", num_gate, "...")
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
    AND = Perceptron(2, bias=-1.0)
    # start training
    print(AND.weights)
    valid_percentage = AND.validate(validate_examples, validate_labels, verbose=True)
    print(valid_percentage)
    i = 0
    while valid_percentage < 0.98: # We want our Perceptron to have an accuracy of at least 98%
        i += 1
        AND.train(training_examples, training_labels, 0.2)  # Train our Perceptron
        print('------ Iteration ' + str(i) + ' ------')
        print(AND.weights)
        valid_percentage = AND.validate(validate_examples, validate_labels, verbose=True) # Validate it
        print(valid_percentage)
    # NOT Gate

    # OR Gate
