import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class MLP:
    def __init__(self, input_size=784, h1=128, h2=64, output_size=10):
        # He başlatma
        self.w1 = np.random.randn(input_size, h1) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(h1)
        self.w2 = np.random.randn(h1, h2) * np.sqrt(2 / h1)
        self.b2 = np.zeros(h2)
        self.w3 = np.random.randn(h2, output_size) * np.sqrt(2 / h2)
        self.b3 = np.zeros(output_size)
    
    def forward(self, x):
        # Girişi düzleştir
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        
        self.z1 = x @ self.w1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = softmax(self.z3)
        return self.a3
    
    def loss(self, y_hat, y):
        m = y.shape[0]
        log_likelihood = -np.log(y_hat[range(m), y] + 1e-15)
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, x, y, lr=0.01):
        m = x.shape[0]
        
        # Girişi düzleştir
        if x.ndim == 3:
            x = x.reshape(m, -1)
        
        # Çıkış katmanı gradyanları
        dz3 = self.a3.copy()
        dz3[range(m), y] -= 1
        dz3 /= m
        
        # 3. katman gradyanları
        dw3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0)
        
        # 2. katman gradyanları
        dz2 = dz3 @ self.w3.T
        dz2[self.a2 <= 0] = 0  # ReLU türevi
        
        dw2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        
        # 1. katman gradyanları
        dz1 = dz2 @ self.w2.T
        dz1[self.a1 <= 0] = 0  # ReLU türevi
        
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        # Ağırlıkları güncelle
        self.w1 -= lr * dw1
        self.b1 -= lr * db1
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w3 -= lr * dw3
        self.b3 -= lr * db3