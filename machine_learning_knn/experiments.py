import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from knn_classifier import KNNClassifier

# === 0. RESULT klasörü oluştur ===
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# === 1. Veri Setini Yükleme ===
print("1. MNIST Digits Veri Seti Yükleniyor...")
digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Eğitim verisi boyutu: {X_train.shape}")
print(f"Test verisi boyutu: {X_test.shape}")

# === 2. Model Eğitimi ===
print("\n2. KNN Model Eğitimi (k=3, L2)")
knn_model = KNNClassifier(k=3, distance_metric='l2')
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)

# === 3. Accuracy Hesaplama Kısmı===
accuracy = knn_model.score(X_test, y_test)
print("-" * 30)
print(f"Test Accuracy: {accuracy:.4f}")
print("-" * 30)

# Accuracy’yi results klasörüne yaz
with open(os.path.join(RESULT_DIR, "accuracy.txt"), "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")

# === 4. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=digits.target_names, yticklabels=digits.target_names)
plt.title('Confusion Matrix (KNN, k=3, L2)')
plt.ylabel('Gerçek Etiket')
plt.xlabel('Tahmin Edilen Etiket')

# Görseli kaydet
cm_path = os.path.join(RESULT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight")
print(f"Confusion matrix '{cm_path}' dosyasına kaydedildi.")
plt.close()

# === 5. Örnek Tahmin Görselleştirmeleri ===
print("\n4. Örnek Tahmin Görselleştirmeleri (İlk 10 test örneği)")
fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for i in range(10):
    image = X_test[i].reshape(8, 8)
    true_label = y_test[i]
    predicted_label = int(y_pred[i])

    axes[i].imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    axes[i].set_title(f"G: {true_label}\nT: {predicted_label}",
                      color='green' if true_label == predicted_label else 'red')
    axes[i].axis('off')

plt.suptitle(f"KNN Tahminleri (k={knn_model.k}, Metrik={knn_model.distance_metric})", y=1.05)

# Görseli kaydet
pred_path = os.path.join(RESULT_DIR, "sample_predictions.png")
plt.savefig(pred_path, bbox_inches="tight")
print(f"Örnek tahmin görseli '{pred_path}' dosyasına kaydedildi.")
plt.close()

print("\n Tüm sonuçlar 'results/' klasörüne kaydedildi.")
