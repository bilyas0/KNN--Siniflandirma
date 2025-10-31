Sıfırdan KNN Sınıflandırıcı (KNN Classifier from Scratch)
Bu proje, K-En Yakın Komşuluk (KNN) algoritmasının NumPy kütüphanesi kullanılarak sıfırdan uygulanmasını içermektedir. Geliştirilen sınıflandırıcı, scikit-learn kütüphanesinden yüklenen MNIST el yazısı rakamları veri setini tanımak amacıyla eğitilmiş ve test edilmiştir.

Projenin temel amacı, KNN algoritmasının iç mekaniklerini anlamak ve model performansını etkileyen k değeri ve mesafe metrikleri gibi hiperparametrelerin etkisini analiz etmektir.

Örnek: k=3 ve L2 mesafesi için oluşturulan karmaşıklık matrisi.

Projenin Özellikleri
Sıfırdan Uygulama: Çekirdek algoritma, scikit-learn gibi hazır kütüphanelere dayanmadan, fit, predict ve score metotları ile kendi KNNClassifier sınıfımızda geliştirilmiştir.

Parametre Değerlendirmesi: Modelin doğruluğu üzerinde kritik etkiye sahip olan farklı parametreler sistematik olarak değerlendirilmiştir:

K Değeri Analizi: Model, k (komşu sayısı) için [1, 3, 5, 7, 9, 11, 15, 21] gibi farklı değerlerle test edilerek en uygun komşu sayısı araştırılmıştır.

Mesafe Metriği Karşılaştırması: L1 (Manhattan) ve L2 (Euclidean) mesafe metriklerinin performansı karşılaştırılmış ve sonuçlar analiz edilmiştir.

Scikit-learn ile Karşılaştırma: Kendi yazdığımız sınıflandırıcının performansı, scikit-learn kütüphanesinin standart KNeighborsClassifier'ı ile karşılaştırılarak kendi uygulamamızın doğruluğu teyit edilmiştir.

Detaylı Görselleştirme: Analiz sonuçlarını daha anlaşılır kılmak için karmaşıklık matrisi (confusion matrix), örnek tahmin görselleri ve parametre analiz grafikleri gibi görselleştirmeler kullanılmıştır.

Proje Yapısı
Odev_Knn/
|
|------ knn_classifier.py         # Sıfırdan geliştirilen KNN sınıfını içerir
|------ main_notebook.py          # Tüm deneylerin, analizlerin ve yorumların yapıldığı ana Jupyter Notebook dosyası
|------ visualization.py          # Sonuçları görselleştirmek için kullanılan fonksiyonları barındırır
|------ experiments.py            # Modelin test edilmesi için basit bir script
|------ README.md                 # Bu dosya
|
|------ results/                  # Modelin çalıştırılmasıyla üretilen tüm çıktıların (grafikler, metinler) kaydedildiği klasör
|       |------ confusion_matrix.png
|       |------ k_value_analysis.png
|       |------ sample_predictions.png
|       |------ comparison_table.png
|       |------ accuracy.txt
Gereksinimler
Projeyi çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyacınız vardır:

numpy

matplotlib

scikit-learn

seaborn

Bu kütüphaneleri pip ile kolayca yükleyebilirsiniz:

Bash

pip install numpy scikit-learn matplotlib seaborn
Kullanım
Tüm adımları, kod açıklamalarını ve çıktıları görmek için main_notebook.py dosyasını bir Jupyter Notebook veya Jupyter Lab ortamında açın ve hücreleri sırasıyla çalıştırın.

Veri Yükleme ve Hazırlama: MNIST veri setini yükler ve eğitim/test olarak böler.

Model Eğitimi ve Test: Kendi KNNClassifier sınıfımızı kullanarak modeli eğitir ve test eder.

Parametre Analizi: Farklı k değerleri ve mesafe metrikleri için testler yapar ve sonuçları görselleştirir.

Sklearn Karşılaştırması: Kendi modelimizin sonuçlarını scikit-learn ile karşılaştırır.

Tüm görsel çıktılar ve analiz sonuçları otomatik olarak results/ klasörüne kaydedilecektir.
