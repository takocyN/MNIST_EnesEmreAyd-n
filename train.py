import numpy as np
from model import MLP



# Dosya isimleri
train_images_file = 'train-images-idx3-ubyte'
train_labels_file = 'train-labels-idx1-ubyte'
test_images_file = 't10k-images-idx3-ubyte'
test_labels_file = 't10k-labels-idx1-ubyte'

# IDX dosya yükleme
def load_images_idx(filename):
    with open(filename, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 28*28) / 255.0

def load_labels_idx(filename):
    with open(filename, 'rb') as f:
        f.read(8)
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Verileri yükle
train_x = load_images_idx(train_images_file)
train_y = load_labels_idx(train_labels_file)
test_x = load_images_idx(test_images_file)
test_y = load_labels_idx(test_labels_file)

print(f"Eğitim görüntüleri: {train_x.shape}")
print(f"Test görüntüleri: {test_x.shape}")

# Model oluştur
model = MLP()
lr = 0.1
epochs = 10000
batch_size = 128

# Eğitim
print(f"(epochs={epochs}, lr={lr})")
for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0
    
    for i in range(0, len(train_x), batch_size):
        xb = train_x[i:i+batch_size]
        yb = train_y[i:i+batch_size]
        
        y_pred = model.forward(xb)
        loss = model.loss(y_pred, yb)
        preds = np.argmax(y_pred, axis=1)
        acc = np.mean(preds == yb) * 100
        
        epoch_loss += loss
        epoch_acc += acc
        batch_count += 1
        model.backward(xb, yb, lr)
    
    avg_loss = epoch_loss / batch_count
    avg_acc = epoch_acc / batch_count
    print(f"Epoch {epoch+1:2d}/{epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")

# Test
test_predictions = []
for i in range(0, len(test_x), batch_size):
    xb = test_x[i:i+batch_size]
    y_pred = model.forward(xb)
    batch_preds = np.argmax(y_pred, axis=1)
    test_predictions.extend(batch_preds)

test_acc = np.mean(np.array(test_predictions) == test_y) * 100
print(f"Test Doğruluğu: {test_acc:.2f}%")

# Modeli kaydet
np.savez('model_weights.npz',
         w1=model.w1, b1=model.b1,
         w2=model.w2, b2=model.b2,
         w3=model.w3, b3=model.b3)
print("=" * 50)