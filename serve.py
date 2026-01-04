import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Aktivasyon fonksiyonlarını tanımla
def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# MLP sınıfını tanımla
class MLP:
    def __init__(self, input_size=784, h1=128, h2=64, output_size=10):
        self.w1 = np.zeros((input_size, h1))
        self.b1 = np.zeros(h1)
        self.w2 = np.zeros((h1, h2))
        self.b2 = np.zeros(h2)
        self.w3 = np.zeros((h2, output_size))
        self.b3 = np.zeros(output_size)
    
    def forward(self, x):
        # Girişi düzleştir
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        
        z1 = x @ self.w1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.w3 + self.b3
        a3 = softmax(z3)
        return a3

# Model yükle
try:
    data = np.load('model_weights.npz')
    model = MLP()
    model.w1 = data['w1']
    model.b1 = data['b1']
    model.w2 = data['w2']
    model.b2 = data['b2']
    model.w3 = data['w3']
    model.b3 = data['b3']
    print("Model başarıyla yüklendi!")
except Exception as e:
    print(f"Model yüklenirken hata oluştu: {e}")
    print("Eğitilmiş model dosyası 'model_weights.npz' mevcut değil.")
    print("Önce modeli eğitmek için train_model.py çalıştırılmalı.")
    exit()

# Veri yükleme
def load_images_idx(filename):
    try:
        with open(filename, 'rb') as f:
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 28 * 28) / 255.0
    except Exception as e:
        print(f"{filename} dosyası yüklenirken hata: {e}")
        return None

def load_labels_idx(filename):
    try:
        with open(filename, 'rb') as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)
    except Exception as e:
        print(f"{filename} dosyası yüklenirken hata: {e}")
        return None

# MNIST test verilerini yükle
print("Test verileri yükleniyor...")
test_x = load_images_idx('t10k-images-idx3-ubyte')
test_y = load_labels_idx('t10k-labels-idx1-ubyte')

if test_x is None or test_y is None:
    print("Test verileri yüklenemedi!")
    print("Aşağıdaki dosyaların mevcut olduğundan emin olun:")
    print("1. t10k-images-idx3-ubyte")
    print("2. t10k-labels-idx1-ubyte")
    exit()

print(f"Test verisi yüklendi: {len(test_x)} örnek")

# Tüm test verisi için tahminler
predictions = []
all_pred_probs = []
batch_size = 100

print("Test verisi üzerinde tahmin yapılıyor...")
for i in range(0, len(test_x), batch_size):
    x_batch = test_x[i:i + batch_size]
    pred_probs = model.forward(x_batch)
    y_pred = np.argmax(pred_probs, axis=1)
    predictions.extend(y_pred)
    all_pred_probs.extend(pred_probs)
    if i % 1000 == 0:
        print(f"  {i}/{len(test_x)} örnek işlendi")

# Doğruluk hesapla
predictions_array = np.array(predictions)
total_accuracy = np.mean(predictions_array == test_y) * 100

# Performans metriklerini hesapla
all_pred_probs_array = np.array(all_pred_probs)

# One-hot encode gerçek etiketler
y_true_onehot = np.zeros((len(test_y), 10))
y_true_onehot[np.arange(len(test_y)), test_y] = 1

# MSE (Mean Squared Error)
mse = mean_squared_error(y_true_onehot, all_pred_probs_array)

# MAE (Mean Absolute Error)
mae = mean_absolute_error(y_true_onehot, all_pred_probs_array)

# R^2 Score
r2 = r2_score(y_true_onehot, all_pred_probs_array)

# Sonuçları yazdır
print("\n" + "="*50)
print("PERFORMANS METRİKLERİ")
print("="*50)
print(f"Ortalama doğruluk: {total_accuracy:.2f}%")
print(f"MSE (Mean Squared Error): {mse:.6f}")
print(f"MAE (Mean Absolute Error): {mae:.6f}")
print(f"R² (R-squared) Score: {r2:.6f}")
print("="*50)

# Rastgele 100 örnek seç
np.random.seed(42)
indices = np.random.choice(len(test_x), 100, replace=False)

# PDF oluştur
print("\nPDF raporu oluşturuluyor...")
with PdfPages('mnist_tahminler.pdf') as pdf:
    for page in range(4):
        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        axes = axes.flatten()

        correct_page = 0
        for i in range(25):
            idx = indices[page * 25 + i]
            x = test_x[idx:idx + 1]
            y_true = test_y[idx]
            y_pred = np.argmax(model.forward(x))

            if y_pred == y_true:
                correct_page += 1

            ax = axes[i]
            ax.imshow(x.reshape(28, 28), cmap='gray')
            ax.axis('off')

            if y_pred == y_true:
                ax.set_title(f"Doğru\nGerçek: {y_true}\nTahmin: {y_pred}", color='green', fontsize=9)
            else:
                ax.set_title(f"Yanlış\nGerçek: {y_true}\nTahmin: {y_pred}", color='red', fontsize=9)

        page_accuracy = (correct_page / 25) * 100
        plt.suptitle(f"Sayfa {page + 1}/4 - Bu sayfa doğruluk: {page_accuracy:.1f}%", fontsize=12, y=0.995)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

print(f"\nmnist_tahminler.pdf başarıyla oluşturuldu!")
print("\nNot: Modelin performansını değerlendirmek için aşağıdaki dosyalar gereklidir:")
print("1. model_weights.npz - Eğitilmiş model ağırlıkları")
print("2. t10k-images-idx3-ubyte - MNIST test görüntüleri")
print("3. t10k-labels-idx1-ubyte - MNIST test etiketleri")
