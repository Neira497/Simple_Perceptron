import numpy as np

X = np.array([[0, 0],   # 0
              [0, 1],   # 0
              [1, 0],   # 0
              [1, 1]])  # 1

Y = np.array([0, 0, 0, 1])

def step_fun(x):
    return np.where(x >= 0, 1, 0)
    
class Perceptron_Simple():
    def __init__(self):
        self.W = np.random.rand(2) * 2 - 1
        self.B = np.random.rand(1) * 2 - 1
        
    def predict(self, x):
        z = np.dot(x, self.W) + self.B
        return step_fun(z)
    
    def fit(self, X, Y, epocas, lr=0.1):
        for epoca in range(epocas):
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                y_pred = self.predict(x)
                error = y_pred - y
                self.W = self.W - lr * error * x
                self.B = self.B - lr * error
    
Neuron = Perceptron_Simple()
Neuron.fit(X, Y, 20)

prueba = np.array([1, 1])
print(Neuron.predict(prueba))
