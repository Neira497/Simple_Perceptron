import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"D:\Neuronal Network\train.csv")    # leemos los datos
data = np.array(data)           # Convertimos los datos en un array
m, n = data.shape               # m tomará el valor de la cantidad de datos
                                # y la n tomará el valor de la cantidad de pixeles
np.random.shuffle(data)         # Hace a los datos aleatorios

data_dev = data[0:1000].T       # Datos de testing del 0 a 1000
Y_dev = data_dev[0]             # Clases
X_dev = data_dev[1:n]
X_dev = X_dev / 255             # Normalizacion de los pixeles

data_train = data[1000:m].T     # Datos de entrenamiento del mil a 42000
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_, m_train = X_train.shape

def init_params():
    W1 = np.random.rand(20, 784) * 2 - 1    # 784 entradas para 10 neuronas
    b1 = np.random.rand(20, 1) * 2 - 1      # 1 sesgo para 10 neuronas
    W2 = np.random.rand(10, 20) * 2 - 1     # 10 entradas para 10 neuronas clasificatorias
    b2 = np.random.rand(10, 1) * 2 - 1      # 1 sesgo para 10 neuronas
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, one_hot_Y

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2, valor = backward_prop(Z1, A1, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 250 == 0:
            print(f"Epoch {i}/{iterations}", end=" ")
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            print("Accuracy: ", str(round(accuracy, 2) * 100) + "%")
    return W1, b1, W2, b2, valor

W1, b1, W2, b2, valor = gradient_descent(X_train, Y_train, 1.2, 20)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_dev[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_dev[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    if(prediction == label):
        word = "Prediccion correcta"
        print(word)
    else:
        word = "Prediccion incorrecta"
        print(word)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation = "nearest")
    plt.show()
    return word
    
# import time
pCorrecta = 0
pIncorrecta = 0
for i in range(len(Y_dev)):
    word = test_prediction(i, W1, b1, W2, b2)
    if i % 10 == 0:
        if word == "Prediccion correcta":
            pCorrecta += 1
        else:
            pIncorrecta += 1
    # time.sleep(0.5)

pCorrecta = pCorrecta / len(Y_dev)
pIncorrecta = pIncorrecta / len(Y_dev)
print(f"Predicciones correctas: {pCorrecta}")
print(f"Predicciones incorrectas: {pIncorrecta}")