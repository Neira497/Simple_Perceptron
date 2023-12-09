import numpy as np

X = np.array([[1, 1],
              [1, 0],
              [0, 1],
              [0, 0]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

class NeuralNet():
    def __init__(self, input_size, hidden_size, output_size, lr):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        
        self.W1 = np.random.rand(input_size, hidden_size) * 2 - 1
        self.b1 = np.random.rand(1, hidden_size) * 2 - 1
        self.W2 = np.random.rand(hidden_size, output_size) * 2 - 1
        self.b2 = np.random.rand(1, output_size) * 2 - 1
        
    def forwardPass(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def predict(self, input):
        return self.forwardPass(input)
    
    def backpropagation(self, X, Y, output):
        error = (output - Y)
        delta_error = error * self.sigmoid_deriv(self.z2)
        
        delta_W2 = np.dot(self.a1.T, delta_error)
        delta_b2 = np.sum(delta_error, axis = 0, keepdims=True)
        
        delta_W1 = np.dot(X.T, (delta_error @ self.W2.T * self.sigmoid_deriv(self.z1)))
        delta_b1 = np.sum(delta_error @ self.W2.T * self.sigmoid_deriv(self.z1), axis = 0)
        
        self.W2 = self.W2 - self.lr * delta_W2
        self.b2 = self.b2 - self.lr * delta_b2
        self.W1 = self.W1 - self.lr * delta_W1
        self.b1 = self.b1 - self.lr * delta_b1
        
    def fit(self, X, Y, epochs):
        for epoch in range(epochs):
            output = self.forwardPass(X)
            self.backpropagation(X, Y, output)
            if epoch % 100 == 0:
                print(f"Error de la red: {self.MSE(Y)}, epoca: {epoch}")
        
    def sigmoid_deriv(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def sigmoid(self, x):
        out = 1 / (1 + np.exp(-x))
        return out
    
    def MSE(self, Y):
        mse = np.mean(1/2 * (self.forwardPass(X) - Y) ** 2)
        return mse
    
if __name__ == "__main__":
    nn = NeuralNet(2, 6, 1, 1.5)
    nn.fit(X, Y, 2000)
    print(nn.predict([[1, 1]]))
    print(nn.predict([[1, 0]]))
    print(nn.predict([[0, 1]]))
    print(nn.predict([[0, 0]]))
    