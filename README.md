🧠 KNN (K-Nearest Neighbors) Görüntü Sınıflandırma Projesi

Bu proje, K-En Yakın Komşuluk (KNN) algoritmasını sıfırdan (NumPy ile) geliştirip,
Scikit-learn versiyonu ile performans karşılaştırması yapar.
MNIST benzeri el yazısı rakamları (Digits) veri seti kullanılmıştır.


📂 Proje Yapısı
📁 KNN_Project
├── knn_classifier.py          # KNN algoritmasının sıfırdan yazıldığı Python sınıfı
├── main.py                    # Ana dosya (veri yükleme, model eğitimi, test, karşılaştırma)
├── visualization.py            # Tüm görselleştirmeleri içeren yardımcı modül
├── results/                    # Sonuç dosyalarının kaydedildiği klasör
│   ├── accuracy.txt
│   ├── confusion_matrix.png
│   ├── sample_predictions.png
│   ├── k_value_analysis.png
│   ├── distance_comparison.png
│   └── comparison_table.png
└── README.md                   # Bu belge

⚙️ Kurulum
Gerekli kütüphaneleri yükleyin:
pip install numpy matplotlib seaborn scikit-learn

🚀 Çalıştırma
Projeyi terminalden başlatın:
python main.py


Program:
Digits veri setini yükler (sklearn.datasets.load_digits()),
Kendi yazdığınız KNNClassifier sınıfını (k=3, L2) eğitir,
Tahminleri yapar ve doğruluğu hesaplar,
Sonuçları results/ klasörüne kaydeder.

🧩 KNNClassifier (knn_classifier.py)
🔹 Önemli Metotlar
Metot	Açıklama
fit(X, y)	Eğitim verilerini saklar
compute_distances(X)	Test ve eğitim örnekleri arasındaki mesafeleri hesaplar
predict(X)	En yakın k komşuya göre tahmin yapar
score(X, y)	Model doğruluğunu hesaplar
🔹 Mesafe Hesaplama Mantığı
diffs = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]


Bu satır, her test örneği ile tüm eğitim örnekleri arasındaki farkları tek seferde hesaplar.
NumPy broadcasting kullanılır → (n_test, n_train, n_features) boyutlu fark matrisi oluşturulur.

🎨 Görselleştirmeler (visualization.py)

Bu dosya, modelin performansını anlamak ve raporlamak için görseller oluşturur.

🔹 1. Confusion Matrix
plot_confusion_matrix(y_test, y_pred, class_names)
Gerçek ve tahmin edilen etiketleri karşılaştırır
Renk yoğunluğu doğru–yanlış tahmin oranını gösterir
Kaydedilen dosya: results/confusion_matrix.png

🔹 2. Örnek Tahmin Görselleri
plot_sample_predictions(X_test, y_test, y_pred)
10 rastgele test örneği gösterir
Doğru tahminler yeşil, yanlışlar kırmızı
Kaydedilen dosya: results/sample_predictions.png

🔹 3. K Değeri Analizi
plot_k_analysis(k_values, accuracies)
Farklı k değerlerinin doğruluğa etkisini gösterir
En iyi k kırmızı noktayla belirtilir
Kaydedilen dosya: results/k_value_analysis.png

🔹 4. L1 – L2 Mesafe Karşılaştırması
plot_distance_comparison(k_values, l1_accuracies, l2_accuracies)
Manhattan (L1) ve Öklid (L2) metriklerini kıyaslar
Kaydedilen dosya: results/distance_comparison.png

🔹 5. Karşılaştırma Tablosu
create_comparison_table(k_values, l1_accuracies, l2_accuracies)
Her k değeri için L1 ve L2 doğruluklarını tablo olarak gösterir
Farkları (L2 - L1) sütununda görüntüler
Kaydedilen dosya: results/comparison_table.png

📊 Elde Edilen Sonuçlar
Deney	Açıklama	Doğruluk
k=3, L2	Temel model	~0.98
k=7, L2	En iyi doğruluk	~0.985
L1 vs L2	Karşılaştırma	L2 genelde daha iyi
sklearn karşılaştırması	KNeighborsClassifier ile	Neredeyse aynı

🧠 Öğrenilenler
K değeri seçimi, modelin başarısını doğrudan etkiler.
L2 (Euclidean) mesafesi genelde daha stabil sonuç verir.
KNN eğitimde hızlı, ama tahminde yavaş bir algoritmadır.
visualization.py ile sonuçların analizi ve raporlaması kolaylaşır.

🏁 Sonuç

Bu proje, KNN algoritmasını derinlemesine anlamak,
NumPy ile sıfırdan uygulamak,
ve scikit-learn sürümüyle kıyaslamak için güçlü bir örnektir.
Tüm grafikler, tablolar ve doğruluk sonuçları results/ klasöründe toplanır.
