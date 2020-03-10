import numpy as np
import matplotlib.pyplot as plt

class DenseNN:
    """
    A NN with simple hyperparameters and visualization extras for teaching.
    """

    def __init__(self, layer_shapes):
        self.layer_shapes = layer_shapes
        self.W = [np.random.randn(cur, nex) for cur, nex in zip(layer_shapes[:-1], layer_shapes[1:])]


    @staticmethod
    def ReLU(x):
        """
        Activation function returns 0 when x < 0 and x when x > 0.
        """
        return np.maximum(x,0)

    @staticmethod
    def ReLU_grad(x):
        """
        Returns 0 when x < 0 and 1 when x > 0.
        """
        return(x>0).astype('float64')

    @staticmethod
    def MSE(out, y):
        """
        The loss function of the network, mean square error.
        """
        n = len(y)
        sq_error = (out - y)**2
        loss = np.sum(sq_error)/n

        return loss

    @staticmethod
    def MSE_grad(out, y):
        """
        The derivative for the mean square error loss function.
        """
        n = len(y)
        grad = (2/n) * (out - y)

        return grad

    def forward_pass(self):
        """
        Calculates the output of the network with the current weights.
        """

        #Initialize A and Z
        self.A = [self.XTr]
        self.Z = [self.XTr]

        for layerW, layerZ in zip(self.W[:-1], self.Z):
            A_next = layerZ @ layerW
            Z_next = self.ReLU(A_next)

            self.A.append(A_next)
            self.Z.append(Z_next)

        A_final = self.Z[-1] @ self.W[-1]

        self.A.append(A_final)
        self.Z.append(A_final)

    def backprop(self):
        """
        Calculates the gradients of the loss for each layer.
        """

        #Convert delta to a row vector to make things easier
        delta = (self.MSE_grad(self.Z[-1].flatten(), self.yTr)).reshape(-1,1)

        #Compute gradients with backprop
        self.gradients = []

        for layer in range(len(self.W)-1, -1, -1):
            grad = (delta.T @ self.Z[layer]).T
            delta = self.ReLU_grad(self.A[layer]) * (self.W[layer] @ delta.T).T

            self.gradients.append(grad)

        self.gradients.reverse()

    def fit(self, XTr, yTr, iters, lr = .0001):
        """
        Fits the model to the given training data.
        """
        self.XTr = XTr
        self.yTr = yTr

        self.losses = np.zeros(iters)
        for i in range(iters):
            #Do a forward pass
            self.forward_pass()
            #Calculate the loss
            #self.losses[i] = self.MSE(self.Z[-1].flatten(), self.yTr)
            #Do backprop
            self.backprop()
            for j in range(len(self.W)):
                self.W[j] -= lr * self.gradients[j]

    def predict(xTe):
        self.A = [self.xTe]
        self.Z = [self.xTe]

        for layerW, layerZ in zip(self.W[:-1], self.Z):
            A_next = layerZ @ layerW
            Z_next = self.ReLU(A_next)

            self.A.append(A_next)
            self.Z.append(Z_next)

        A_final = self.Z[-1] @ self.W[-1]

        self.A.append(A_final)
        self.Z.append(A_final)

        return self.Z[-1]

def ground_truth(x):
    return(x**2 + 10 * np.sin(x))

def generate_data():
    x = np.arange(0,5,0.1)
    y = ground_truth(x)
    x2d = np.concatenate([x, np.ones(x.shape)]).reshape(2, -1).T

    return x2d, y
