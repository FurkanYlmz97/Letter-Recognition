__author__ = "Furkan Yılmaz"

import numpy as np
from matplotlib import pyplot as plt


def sigmoid_activation_function(input):
    return 1 / (1 + np.exp(-input))


def training(train_images, train_labels, weights, bias_terms, learning_rate, show_weight_images=False, show_rmse=False,
             color_input='b', return_weights=False):
    # output vector of the neuron
    outputs = np.zeros(26, float)

    # mse in one single  iteration
    single_mse = []

    # mse (take the mean of the previous single mses'
    NN_mse = []

    # 10000 iterations
    for i in range(10000):

        # Select a random image as input
        selected_input_number = np.random.random_integers(0, 5199, 1)
        input = train_images[selected_input_number]

        # standardize the input
        input = input / 255

        # the label of the input and the desired output vector
        input_label = train_labels[selected_input_number]
        desired_output_vector = np.zeros(26)
        desired_output_vector[input_label - 1] = 1

        # turn the matrix input to an array
        input_flatten = input.flatten()

        # Finding the mse value
        mse = 0
        for m in range(26):
            outputs[m] = sigmoid_activation_function(np.dot(weights[m], input_flatten) + bias_terms[m])
            mse = mse + (outputs[m] - desired_output_vector[m]) * (outputs[m] - desired_output_vector[m]) / 2
        mse = mse / 26
        single_mse.append(mse)

        # Networks mse value is recorded in this array in every iteration
        NN_mse.append(np.mean(single_mse))

        # Gradient descent matrix for weights
        gradient_descent_weights = np.outer(
            np.multiply((outputs - desired_output_vector), np.multiply(outputs, (1 - outputs))), input_flatten)

        # Gradient descent vector for bias values
        gradient_descent_bias = np.multiply((desired_output_vector - outputs), np.multiply(outputs, (1 - outputs)))

        # updating the weights and bias terms with
        weights = weights - learning_rate * gradient_descent_weights
        bias_terms = bias_terms - learning_rate * gradient_descent_bias

    # Reshape the weights to show them as images
    neuron_weights = np.reshape(weights, (26, 28, 28))

    # returns the mse error of NN after all the iterations
    print("MSE: " + str(np.around(NN_mse[len(NN_mse) - 1], 5)))

    # show the weights as images
    if show_weight_images:
        plt.figure()
        # Show the weights as images
        for l in range(26):
            plt.subplot(7, 4, l + 1)
            plt.title("neuron # : " + str(l))
            plt.axis("off")
            plt.tight_layout()
            plt.imshow(neuron_weights[l])

    # plots the mse values
    if show_rmse is True:
        plt.figure()
        plt.plot(NN_mse, color=str(color_input), label="LR:" + str(learning_rate))
        plt.title("MSE Value in every Iteration")
        plt.ylabel("MSE Value")
        plt.xlabel("Iteration Number")

    if return_weights:
        return weights, bias_terms


def test(test_images, test_labels, weights, bias_terms):
    # Mse value of the NN
    NN_mse = []

    # This is a dummy variable will be used to compute the mse
    single_mse = []

    # The number of times that the NN classified successfully
    true_number = 0

    # Test all images
    for i in range(len(test_images)):

        # Take the images one by one and standardize them
        input = test_images[i]
        input = input / 255

        # Create desired output
        input_label = test_labels[i]
        desired_output_vector = np.zeros(26)
        desired_output_vector[input_label - 1] = 1

        # Convert input to a 1D array
        input = input.flatten()

        # Some initializations
        outputs = np.zeros(26)
        mse = 0
        max = 0
        max_label = 1

        # For all 26 output neurons find the output vector mse and the winner neuron
        for m in range(26):
            outputs[m] = sigmoid_activation_function(np.dot(weights[m], input) + bias_terms[m])
            mse = mse + (outputs[m] - desired_output_vector[m]) * (outputs[m] - desired_output_vector[m]) / 2

            if outputs[m] > max:
                max = outputs[m]
                max_label = m + 1

        if max_label == input_label[0]:
            true_number += 1

        # Find the NN's mse
        mse = mse / 26
        single_mse.append(mse)
        NN_mse.append(np.mean(single_mse))

    # Shows the performance
    print("Performance %", end="")
    print(round((true_number / len(test_images)) * 100))

    # Shows the NN's mse value
    print("Test MSE: ", end="")
    return np.around(NN_mse[len(NN_mse) - 1], 5)


if __name__ == '__main__':

    # Load the dataset
    test_images = np.load("test_images.npy")
    test_labels = np.load("test_labels.npy")
    train_images = np.load("train_images.npy")
    train_labels = np.load("train_labels.npy")

    # change axis for ease
    test_images = np.swapaxes(test_images, 0, 2)
    test_images = np.swapaxes(test_images, 1, 2)
    train_images = np.swapaxes(train_images, 0, 2)
    train_images = np.swapaxes(train_images, 1, 2)

    # weights are initially determined by a gaussian
    weights = np.zeros((26, 28 * 28))
    for i in range(26):
        for m in range(28 * 28):
            weights[i][m] = np.random.normal(0, 0.1)

    # bias terms are initially gaussian distributed
    bias_terms = np.random.normal(0, 0.1, 26)

    # Show the lowest mse valued NN's weights as a image
    trained_weights, trained_biases = training(train_images, train_labels, weights, bias_terms, 0.19, True, True, 'r', True)
    print("Test Results of the NN")
    print(test(test_images, test_labels, trained_weights, trained_biases))
    plt.show()
