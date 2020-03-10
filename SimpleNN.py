#A simple nn object used specifically for training purposes.

import numpy as np
import matplotlib.pyplot as plt

class DenseNN:
    """
    A neural network object with simple parameters, intuitive methods and attributes, used for teaching.
    """
    def __init__(self, layer_shapes, max_iters = 10000, tolerance = .0001):
        self.layer_shapes = layer_shapes
        self.W = [np.random.randn(cur, nex) for cur, nex in zip(layer_shapes[:-1], layer_shapes[1:])]
        self.loss = np.inf
        self.max_iters = max_iters
        self.tolerance = tolerance


    def ReLU(x):
        """
        Activation function that returns 0 when input is < 0 and the input when the input is > 0.
        """
        return np.maximum(x,0)

    def ReLU_grad(x):
        #Returns the gradient of ReLU
        #0 if input is < 0 & 1 if input > 0
        return (x> 0).astype('float64')

    def MSE(out,y):
        """
        The loss function for this network, mean square error.
        """
        n = len(y)
        sq_error = (out - y)**2
        loss = np.sum(sq_error)/n

        return loss

    def MSE_grad(out, y):
        """
        The derivative for the mean square error function.
        """
        n = len(y)
        grad = (2/n) * (out - y)

        return grad

    def forward_pass(W, xTr):
        """
        Calculates the output of the network with the current weights.
        """

        #Initialize A and Z
        A = [xTr]
        Z = [xTr]

        for layerW, layerZ in zip(W[:-1], Z):
            A_next = layerZ @ layerW
            Z_next = self.ReLU(A_next)

            A.append(A_next)
            Z.append(Z_next)

        A_final = Z[-1] @ W[-1]

        A.append(A_final)
        Z.append(Z_final)

        return A, Z

    def backprop(W, A, Z, y):
        """
        Calculates the gradients of the loss for each layer.
        """

        #Convert delta to a row vector to make things easier
        delta = (MSE_grad(Z[-1].flatten(), y)).reshape(-1,1)

        #Compute gradients with backprop
        gradients = []

        #This needs to be improved it is very sloppy
        for layer in range(len(W)-1, -1, -1):
            grad = (delta.T @ Z[layer]).T
            delta = self.ReLU_grad(A[layer]) * (W[layer] @ delta.T).T

            gradients.append(grad)

        gradients.reverse()

        return gradients

    def fit(self, XTr, yTr):
        self.losses = np.zeros(self.max_iters)

        # Start training
        for i in range(self.max_iters):

            # Do a forward pass
            A, Z = self.forward_pass(self.W, XTr)

            # Calculate the loss
            losses[i] = self.MSE(Z[-1].flatten(), yTr)

            # Calculate the loss using backprop
            gradients = self.backprop(A, Z, yTr)

            # Update the parameters
            for j in range(len(self.W)):
                self.W[j] -= lr * gradients[j]
        t1 = time.time()
        print('Elapsed time: %.2fs' % (t1-t0))

    def predict(self, X):
        return self.forward_pass(X)

    def score(self, XTe, yTe):
        pass


def ground_truth(x):
    return(x**2 + 10 * np.sin(x))

def generate_data():
   #training data
    x = np.arange(0, 5, 0.1)
    y = ground_truth(x)
    x2d = np.concatenate([x, np.ones(x.shape)]).reshape(2, -1).T

    return x2d, y


