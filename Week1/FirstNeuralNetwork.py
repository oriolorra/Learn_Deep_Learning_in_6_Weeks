import numpy as np

class NeuralNetwork():
    def __init__(self):
        #Seed the random function to get always the same numbers
        np.random.seed(1)

        #Model of 1 neuron by layer
        #first neuron with 5 inputs and 4 outputs, second neuron with 4 inputs and 1 output
        # synapses weights, weigths created between[-1,1] and mean 0
        self.syn0 = 2 * np.random.random((5, 4)) - 1
        self.syn1 = 2 * np.random.random((4, 3)) - 1
        self.syn2 = 2 * np.random.random((3, 1)) - 1

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_deriv(self, x):
        return x * (1 - x)

    #The neural networks think
    def thinks(self, inputs):
        #Pass input through neural network
        l1 = self.__sigmoid(np.dot(inputs, self.syn0))
        l2 = self.__sigmoid(np.dot(l1, self.syn1))
        l3 = self.__sigmoid(np.dot(l2, self.syn2))
        return l1, l2,l3

    def train(self, training_set_input, training_set_output, iterations):
        for j in range(iterations):
            #Pass inputs through the network
            l0 = training_set_input
            l1, l2, l3 = self.thinks(training_set_input)

            #Calculate error
            l3_error = training_set_output - l3

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            l3_delta = l3_error * self.__sigmoid_deriv(l3)

            l2_error = l3_delta.dot(self.syn2.T)
            l2_delta = l2_error * self.__sigmoid_deriv(l2)

            l1_error = l2_delta.dot(self.syn1.T)
            l1_delta = l1_error * self.__sigmoid_deriv(l1)

            # update synapses weight
            self.syn2 += l2.T.dot(l3_delta)
            self.syn1 += l1.T.dot(l2_delta)
            self.syn0 += l0.T.dot(l1_delta)
            print ("Error:" + str(np.mean(np.abs(l3_error))), end='\r')


if __name__ == "__main__":

    # Intialise a single neuron neural network.
    print("The aim of this neural network is to count how many '1' are in the array.\n"
          "And give '0' value if the count is odd or vice versa.")

    neural_network = NeuralNetwork()

    # input data
    training_set_input = np.array( [[0, 0, 1, 0, 0],[0, 1, 0, 1, 0],[0, 1, 0, 0, 1],[0, 1, 1, 0, 1],[1, 1, 1, 0, 0],
                                    [0, 1, 1, 0, 0],[1, 1, 1, 0, 1],[0, 0, 0, 0, 1],[0, 0, 1, 0, 1],[0, 1, 1, 0, 0],
                                    [1, 1, 1, 0, 1], [1, 0, 1, 0, 1],[1, 0, 1, 0, 0],[1, 1, 0, 1, 1]])

    training_set_output = np.array([[0],[1],[1],[0],[0],
                                    [1],[1],[1],[0],[1],
                                    [1],[0],[1],[1]])
    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_input, training_set_output, 100000)

    # Test the neural network with a new situation.
    test = np.array([1, 0, 0, 1, 0])
    print ("\nConsidering new situation " + str(test) + " -> ?: ")
    l1, l2, l3  = neural_network.thinks(test)
    print (l3)

    print("The test results might be not very good because the dataset of train is very small.\n "
          "This Neural Network have 2 hidden layer(5->4->4->1), so this factor affect to learning process too.")

