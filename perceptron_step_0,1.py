import numpy as np
import matplotlib.pyplot as plt
               
data = np.array([[0.5, 0.8],  # 1
                 [0.3, 0.6],  # 0
                 [0.9, 0.7],  # 1
                 [0.2, 0.4],  # 0
                 [0.6, 0.2],  # 0
                 [0.4, 0.5]]) # 0

def Umbral(x):
    return np.where(x >= 0, 1, 0)
    
def ECM(y_true, y_pred):
    return 0.5 * (y_pred - y_true) ** 2

def div_ECM(y_true, y_pred):
    return (y_pred - y_true)
    
class Perceptron():
    def __init__(self):
        self.bias = np.random.rand(1) * 2 - 1
        self.weights = np.random.rand(2) * 2 - 1
        
    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return Umbral(z)
    
    def fit(self, epochs, lr, X, Y):
        for epoch in range(epochs):
            
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                pred = self.predict(x)
                
                errorTotal = ECM(y, pred)
                error = div_ECM(y, pred)
                
                self.weights = self.weights - (lr * error * x)
                self.bias = self.bias - (lr * error)
            if epoch % 5 == 0:
                print(f"Error del perceptron {epoch}: {errorTotal}")    
                print(f"Pesos {epoch}: {self.weights}")
                print(f"Bias {epoch}: {self.bias}")
                print()
 
y_data = np.array([[1], 
                   [0], 
                   [1], 
                   [0], 
                   [0], 
                   [0]])

                        # GRAFICA

plt.scatter(data[y_data[:, 0] == 0].T[0],
            data[y_data[:, 0] == 0].T[1],
            marker = ".", color="red",
            linewidths=5)

plt.scatter(data[y_data[:, 0] == 1].T[0],
            data[y_data[:, 0] == 1].T[1],
            marker = ".", color="blue",
            linewidths=5)

perceptron = Perceptron()
perceptron.fit(50, 0.1, data, y_data)

xlim = plt.xlim()
x_values = np.linspace(xlim[0], xlim[1], 100)
y_values = -(perceptron.weights[0]/perceptron.weights[1]) * x_values - (perceptron.bias[0] / perceptron.weights[1])

plt.plot(x_values, y_values, color="black", linewidth=5)
plt.scatter([0.8], [0.9], color="green", marker = ".", linewidths=5)
plt.grid(True)
plt.show()
