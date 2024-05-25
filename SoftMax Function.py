import numpy as np
import matplotlib.pyplot as plt

def sigmoide(x):
    return 1 / (1 + np.exp(-x))
def step(x):
    return np.heaviside(x,1)
def lineal(x):
    return x
def ReLU(x):
    return np.maximum(0, x)
def Leaky_ReLU(x, alfa=0.1):
    return max(x, alfa * x)
def tangente(x):
    return np.tanh(x)
def soft_max(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)

x = np.linspace(-10, 10, 100)

# y = [sigmoide(i) for i in x]
# y = [step(i) for i in x]
# y = [lineal(i) for i in x]
# y = [ReLU(i) for i in x]
# y = [Leaky_ReLU(i) for i in x]
# y = [tangente(i) for i in x]
y = soft_max(x)

plt.plot(x, y)
plt.xlabel("Eje X")
plt.ylabel("Eje Y")

# plt.title("Función sigmoide")
# plt.title("Función step")
# plt.title("Función lineal")
# plt.title("Función Unidad de Rectificación lineal")
# plt.title("Función Leaky ReLU")
# plt.title("Función tangente hiperbolica")
plt.title("Función soft max")

plt.grid(True)
plt.show()