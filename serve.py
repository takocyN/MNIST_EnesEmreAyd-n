import numpy as np
from model import MLP
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Model yükle
try:
    data = np.load('model_weights.npz')
    model = MLP()
    model.w1, model.b1 = data['w1'], data['b1']
    model.w2, model.b2 = data['w2'], data['b2']
    model.w3, model.b3 = data['w3'], data['b3']
except:
    exit()

# Veri yükleme
def load_images_idx(filename):
    with open(filename, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 28*28) / 255.0

def load_labels_idx(filename):
    with open(filename, 'rb') as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)

test_x = load_images_idx('t10k-images-idx3-ubyte')
test_y = load_labels_idx('t10k-labels-idx1-ubyte')

# Tüm test verisi için doğruluk
predictions = []
batch_size = 100
for i in range(0, len(test_x), batch_size):
    x_batch = test_x[i:i+batch_size]
    y_pred = np.argmax(model.forward(x_batch), axis=1)
    predictions.extend(y_pred)

total_accuracy = np.mean(np.array(predictions) == test_y) * 100


np.random.seed(42)
indices = np.random.choice(len(test_x), 100, replace=False)


with PdfPages('mnist_tahminler.pdf') as pdf:
    for page in range(4):
        fig, axes = plt.subplots(5, 5, figsize=(15, 15))
        axes = axes.flatten()
        
        correct_page = 0
        for i in range(25):
            idx = indices[page * 25 + i]
            x = test_x[idx:idx+1]
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
        plt.suptitle(f"Sayfa {page+1}/4 - Bu sayfa doğruluk: {page_accuracy:.1f}%", fontsize=12, y=0.995)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

print(f"mnist_tahminler.pdf oluşturuldu")
print(f"Ortalama doğruluk: {total_accuracy:.1f}%")