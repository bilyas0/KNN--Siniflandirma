# KNN Görüntü Sınıflandırma Ödevi - Ana Notebook
# Öğrenci Adı: Burak İlyas Şahin
# Öğrenci No: 230212039
# Tarih: 24.10.2025

# %% [markdown]
# # 1. Kütüphanelerin Yüklenmesi
# Gerekli olan Python kütüphanelerini projemize dahil ediyoruz.

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import time
import os

# Kendi yazdığımız yardımcı modülleri import edelim
from knn_classifier import KNNClassifier
from visualization import *

# %%
# Sonuçların kaydedileceği 'results' klasörünü oluşturalım (eğer yoksa)
os.makedirs('results', exist_ok=True)

# %% [markdown]
# # 2. Veri Setinin Yüklenmesi ve Hazırlanması

# %%
# MNIST el yazısı rakamları veri setini yükleyelim
digits = load_digits()
X, y = digits.data, digits.target # X: resim pikselleri, y: etiketler (rakamlar)

# Veri setini eğitim (%80) ve test (%20) olarak ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, )


# %% [markdown]
# ## 2.1 Veri Seti Hakkında Bilgiler
# Yüklediğimiz veri setinin boyutlarını ve özelliklerini inceleyelim.

# %%
# Veri seti boyutlarını ve temel bilgileri yazdır
print("VERİ SETİ BİLGİLERİ")
print("="*10)
print(f"Eğitim Örneği Sayısı: {X_train.shape[0]}")
print(f"Test Örneği Sayısı: {X_test.shape[0]}")
print(f"Özellik (Feature) Sayısı: {X_train.shape[1]} (8x8 piksel)")
print(f"Sınıf Sayısı: {len(np.unique(y_train))} (0-9 arası rakamlar)")
print(f"Eğitim Seti Boyutu: {X_train.shape}")
print(f"Test Seti Boyutu: {X_test.shape}")

# %% [markdown]
# # 3. Görev 1.2: MNIST Veri Seti ile Modelin Test Edilmesi

# %% [markdown]
# ## 3.1 Modelin Eğitilmesi (k=3, L2 Mesafesi)
# Kendi yazdığımız KNNClassifier sınıfını kullanarak modeli eğiteceğiz.

# %%
print("Görev 1.2: Temel KNN Modeli (k=3, L2 - Öklid Mesafesi)")
print("-"*5)

# KNN modelini k=3 ve mesafe metriği 'l2' (Öklid) olarak oluşturalım
knn_model = KNNClassifier(k=3, distance_metric='l2')
start_time = time.time()
knn_model.fit(X_train, y_train)
fit_time = time.time() - start_time
print(f"Eğitim tamamlandı! (Süre: {fit_time:.4f} saniye)")

# %% [markdown]
# ## 3.2 Test Doğruluğunun Hesaplanması
# Eğittiğimiz modelin test verisindeki performansını ölçelim.

# %%
start_time = time.time()
y_pred = knn_model.predict(X_test)
pred_time = time.time() - start_time

# Doğruluk (accuracy) skorunu hesaplayalım
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Tahmin işlemi tamamlandı! (Süre: {pred_time:.4f} saniye)")
print(f"\n{'-'*50}")
print(f"TEST DOĞRULUĞU: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"{'-'*50}\n")

# %% [markdown]
# ## 3.3 Karmaşıklık Matrisi (Confusion Matrix)
# Modelin hangi sınıfları doğru, hangilerini yanlış tahmin ettiğini görsel olarak inceleyelim.

# %%
print("Karmaşıklık matrisi oluşturuluyor...")
class_names = [str(i) for i in range(10)]
plot_confusion_matrix(y_test, y_pred, class_names,
                      save_path='results/confusion_matrix_k3_l2.png')
print("Karmaşıklık matrisi 'results/confusion_matrix_k3_l2.png' olarak kaydedildi.\n")

# %% [markdown]
# ## 3.4 Örnek Tahminlerin Görselleştirilmesi
# Modelin yaptığı tahminlerden bazı örnekleri görelim.

# %%
print("Örnek tahminler görselleştiriliyor...")
plot_sample_predictions(X_test, y_test, y_pred, n_samples=10,
                        save_path='results/sample_predictions_k3_l2.png')
print("Örnek tahminler 'results/sample_predictions_k3_l2.png' olarak kaydedildi.\n")

# %% [markdown]
# # 4. Görev 1.3a: K Değerinin Analizi
# Model performansının farklı 'k' (komşu sayısı) değerlerine göre nasıl değiştiğini analiz edelim.

# %%
print("\n" + "="*50)
print("Görev 1.3a: K Değeri Analizi (L2 Mesafesi)")
print("="*50)

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
l2_accuracies = []

print("\nFarklı k değerleri için model test ediliyor...\n")

for k in k_values:
    print(f"k={k} değeri test ediliyor...", end=" ")

    # Modeli oluştur ve eğit
    knn = KNNClassifier(k=k, distance_metric='l2')
    knn.fit(X_train, y_train)

    # Doğruluk skorunu hesapla ve listeye ekle
    accuracy = knn.score(X_test, y_test)
    l2_accuracies.append(accuracy)

    print(f"Doğruluk: {accuracy:.4f}")

print("\nTüm k değerleri test edildi!")

# %% [markdown]
# ## 4.1 K Değeri Analiz Sonuçlarının Görselleştirilmesi

# %%
print("\nK değeri analiz grafiği oluşturuluyor...")
plot_k_analysis(k_values, l2_accuracies,
                save_path='results/k_value_analysis.png')
print("K değeri analiz grafiği 'results/k_value_analysis.png' olarak kaydedildi.")

# En iyi sonucu veren k değerini bul ve yazdır
best_k_idx = np.argmax(l2_accuracies)
best_k = k_values[best_k_idx]
best_accuracy = l2_accuracies[best_k_idx]

print(f"\n{'='*50}")
print(f"EN İYİ PERFORMANS VEREN K DEĞERİ: {best_k}")
print(f"Bu k değeri ile elde edilen doğruluk: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"{'='*50}\n")

# %% [markdown]
# ## 4.2 K Değeri Analizi - Yorumlar
#
# **Sorular:**
# - Hangi k değeri en iyi sonucu veriyor?
# - K değeri arttıkça doğruluk (accuracy) nasıl değişiyor?
# - Aşırı öğrenme (overfitting) veya eksik öğrenme (underfitting) gözlemlediniz mi?
#
# Cevaplar:
# - En iyi k değeri 7 dir.
# - Düşük k değerleri (örneğin k=1) modelin eğitim verisini ezberlemesine, yani overfittinge yol açabilir. K değeri arttıkça model daha genel bir yapıya bürünür ve genelleme yeteneği artar. Böylece overfiting yapmaktan kaçınırız.
# - Ancak, çok yüksek k değerleri modelin detayları kaçırmasına ve underfitting yapmasına neden olabilir. Optimal k değeri, veri setinin yapısına bağlı olarak bu iki durum arasında bir denge noktasıdır. Genellikle doğruluk skoru belirli bir k değerine kadar artar ve sonra düşmeye başlar.

# %% [markdown]
# # 5. Görev 1.3b: Mesafe Metriklerinin Karşılaştırılması
# L1 (Manhattan) ve L2 (Öklid) mesafelerinin model performansı üzerindeki etkisini karşılaştıralım.

# %%
print("\n" + "="*50)
print("Görev 1.3b: Mesafe Metriği Karşılaştırması (L1 vs L2)")
print("="*50)

l1_accuracies = []

print("\nL1 (Manhattan) mesafe metriği ile test ediliyor...\n")

for k in k_values:
    print(f"k={k} için L1 metriği test ediliyor...", end=" ")

    # L1 mesafe ile model oluştur ve eğit
    knn_l1 = KNNClassifier(k=k, distance_metric='l1')
    knn_l1.fit(X_train, y_train)

    # Doğruluk skorunu hesapla
    accuracy = knn_l1.score(X_test, y_test)
    l1_accuracies.append(accuracy)

    print(f"Doğruluk: {accuracy:.4f}")

print("\nL1 ve L2 karşılaştırması tamamlandı!")

# %% [markdown]
# ## 5.1 Karşılaştırma Grafiği ve Tablosu

# %%
print("\nKarşılaştırma grafikleri oluşturuluyor...")

# Karşılaştırma grafiğini çiz ve kaydet
plot_distance_comparison(k_values, l1_accuracies, l2_accuracies,
                         save_path='results/distance_comparison.png')
print("Karşılaştırma grafiği 'results/distance_comparison.png' olarak kaydedildi.")

# Karşılaştırma tablosunu oluştur ve kaydet
create_comparison_table(k_values, l1_accuracies, l2_accuracies,
                        save_path='results/comparison_table.png')
print("Karşılaştırma tablosu 'results/comparison_table.png' olarak kaydedildi.")

# Sonuçları ekrana yazdır
print("\n" + "="*53)
print("MESAFE METRİĞİ KARŞILAŞTIRMA SONUÇLARI")
print("="*53)
print(f"{'K Değeri':<12} {'L1 Doğruluk':<17} {'L2 Doğruluk':<17} {'Fark (L2-L1)':<17}")
print("-"*53)

for i, k in enumerate(k_values):
    diff = l2_accuracies[i] - l1_accuracies[i]
    print(f"{k:<12} {l1_accuracies[i]:<17.4f} {l2_accuracies[i]:<17.4f} {diff:<+17.4f}")

# Her iki metrik için en iyi sonuçları bul
best_l1_idx = np.argmax(l1_accuracies)
best_l2_idx = np.argmax(l2_accuracies)

print("\n" + "-"*53)
print(f"L1 için en iyi sonuç: k={k_values[best_l1_idx]}, Doğruluk={l1_accuracies[best_l1_idx]:.4f}")
print(f"L2 için en iyi sonuç: k={k_values[best_l2_idx]}, Doğruluk={l2_accuracies[best_l2_idx]:.4f}")
print("="*53 + "\n")

# %% [markdown]
# ## 5.2 Mesafe Metriği Analizi - Yorumlar
#
# **Sorular:**
# - Hangi mesafe metriği genel olarak daha iyi performans gösteriyor?
# - İki metrik arasındaki performans farkları anlamlı mı?
# - Neden bu farklar oluşuyor olabilir?
#
# Cevaplar:
# - L2 metriği bize daha iyi sonuçlar verdi fakat maliyet olarak deaha fazla maliyeti vardır.
# - L1 (Manhattan) mesafesi, farkların mutlak değerlerinin toplamını alır. L2 (Öklid) mesafesi ise farkların karelerinin toplamının karekökünü alır.
# - L2, büyük farklarda karesi alındığından fazlalaştırır. MNIST gibi görüntü verilerinde pikseller arasında önemli olduğundan, iki nokta arasındaki en kısa mesafeyi temsil eden L2 metriğinin genellikle verinin genel yapısını daha iyi yakalaması ve bu nedenle biraz daha iyi sonuç vermesi beklenir.

# %% [markdown]
# # 6. Görev 2: Scikit-learn Kütüphanesi ile Karşılaştırma
# Kendi yazdığımız KNN sınıfının performansını, endüstri standardı olan Scikit-learn kütüphanesindeki implementasyon ile karşılaştıralım.

# %%
print("\n" + "="*50)
print("Görev 2: Scikit-learn KNN ile Karşılaştırma")
print("="*50)

# Scikit-learn KNN modelini oluşturalım (k=3, Öklid mesafesi)
sklearn_knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean') # 'euclidean' L2 demektir
sklearn_knn.fit(X_train, y_train)
sklearn_accuracy = sklearn_knn.score(X_test, y_test)

# Kendi KNN modelimiz (k=3, L2 mesafesi)
your_knn = KNNClassifier(k=3, distance_metric='l2')
your_knn.fit(X_train, y_train)
your_accuracy = your_knn.score(X_test, y_test)

print(f"Scikit-learn KNN Doğruluğu: {sklearn_accuracy:.4f}")
print(f"Kendi KNN Modelimizin Doğruluğu: {your_accuracy:.4f}")
print(f"Doğruluk Farkı: {abs(sklearn_accuracy - your_accuracy):.6f}")
print("="*50 + "\n")

# %% [markdown]
# ## 6.1 Scikit-learn Karşılaştırması - Yorumlar
#
# **Sorular:**
# - Elde ettiğiniz sonuçlar Scikit-learn sonuçları ile benzer mi?
# - Eğer küçük bir fark varsa nedeni ne olabilir?
# - Scikit-learn kullanmanın avantajları nelerdir?
#
# **Cevaplarınız:**
# - Doğruluk değerleri aynıdır.
# - Olası farkların nedeninin başında bence fonksiyonun yanlış bir biçimde yazılması olacaktır. Onun dıında hassas rakamlarla veri hesaplaması olabilir.
# - Scikit learn uzun zamanlı olduğu için hızlı çalışır. Bizim yazdığımız basit kod gibi maliyetli olmaz veri yapıları sağlam olduğundan hızlı çalışır.

# %% [markdown]
# # 7. Genel Sonuçlar ve Öğrenilenler

# %%
print("\n" + "="*50)
print("GENEL SONUÇLAR VE ÖZET")
print("="*50)
print("\n1. K DEĞERİ ANALİZİ:")
print(f"   - Test edilen k değerleri: {k_values}")
print(f"   - En iyi performansı veren k değeri: {best_k}")
print(f"   - Bu k değeri ile en yüksek doğruluk: {best_accuracy:.4f}")

print("\n2. MESAFE METRİĞİ KARŞILAŞTIRMASI:")
best_l1 = max(l1_accuracies)
best_l2 = max(l2_accuracies)
print(f"   - L1 (Manhattan) ile en yüksek doğruluk: {best_l1:.4f}")
print(f"   - L2 (Öklid) ile en yüksek doğruluk: {best_l2:.4f}")
print(f"   - En iyi sonuçlar arasındaki fark: {abs(best_l2-best_l1):.4f}")

print("\n3. SCİKİT-LEARN KARŞILAŞTIRMASI (k=3, L2 için):")
print(f"   - Kendi KNN modelimizin doğruluğu: {your_accuracy:.4f}")
print(f"   - Scikit-learn KNN doğruluğu: {sklearn_accuracy:.4f}")
print(f"   - Aradaki fark: {abs(sklearn_accuracy - your_accuracy):.6f}")

print("\n4. OLUŞTURULAN GÖRSEL DOSYALARI:")
print("   - results/confusion_matrix_k3_l2.png")
print("   - results/sample_predictions_k3_l2.png")
print("   - results/k_value_analysis.png")
print("   - results/distance_comparison.png")
print("   - results/comparison_table.png")


# %% [markdown]
# # 7.1 Öğrenilenler ve Çıkarımlar
#
# Önemli Çıkarımlar ve Öğrenimler:
#
# 1.  K Değerinin Etkisi:
#     - KNN algorıtmasındaki k hiperparetmemiz bizim için çok önemlidir. K değerini ayarlarken ne fazla ne az şekilde ayarlamalıyız.
#     - Overfitting ve underfitting i önlemek için çapraz doğrulama gibi algoritmalar kullanabiliriz.
#
# 2.  Mesafe Metriklerinin Önemi
#     - L1 (Manhattan) ve L2 (Öklid) metrikleri, farklı metrikler farklı sonuçlar veryor.
#
# 3.  KNN Algoritmasının Temel Özellikleri
#     - KNN, eğitim aşamasında bir model inşa etmez, sadece veri setini hafızasında tutar. Bu nedenle eğitim süresi neredeyse sıfırdır.
#     - Tahmin aşaması, yeni bir noktanın tüm eğitim verilerine olan uzaklığını hesaplamayı gerektirdiği için maliyetlidir. Eğitim seti büyüdükçe tahmin süresi de doğru orantılı olarak artar.
#     - Scikit-learn gibi kütüphaneler, bu hesaplama yükünü azaltmak için `Ball Tree` veya `KD Tree` gibi optimize edilmiş veri yapıları kullanır.
#
# 4.  Uygulamaya Yönelik Çıkarımlar
#     - En iyi hiperparametreleri (k değeri, mesafe metriği) bulmak için her zaman çapraz doğrulama gibi algoritmalar kullanılmalıdır.