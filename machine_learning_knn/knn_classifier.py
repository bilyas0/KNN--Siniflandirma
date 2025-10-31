# Dosya Adı: knn_classifier.py

import numpy as np
from collections import Counter

class KNNClassifier:
    """
    K-En Yakın Komşuluk (K-Nearest Neighbors) Sınıflandırıcı
    """
    
    def __init__(self, k=3, distance_metric='l2'):
        """
        KNN sınıflandırıcıyı başlatır
        
        Parameters:
        -----------
        k : int
            Komşu sayısı
        distance_metric : str
            Mesafe metriği ('l1' veya 'l2')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Training verisini kaydeder
        
        Parameters:
        -----------
        X : numpy array, shape (n_samples, n_features)
            Training özellikleri
        y : numpy array, shape (n_samples,)
            Training etiketleri
        """
        # TODO: Training verisini self.X_train ve self.y_train'e kaydedin
        self.X_train = X
        self.y_train = y
    
    def compute_distances(self, X):
        """
        Test örnekleri ile training örnekleri arasındaki mesafeleri hesaplar
        
        Parameters:
        -----------
        X : numpy array, shape (n_test, n_features)
            Test örnekleri
            
        Returns:
        --------
        distances : numpy array, shape (n_test, n_train)
            Mesafe matrisi
        """
        # TODO: Mesafe hesaplama implementasyonu
        if self.X_train is None:
            raise RuntimeError("Model 'fit' edilmedi. Lütfen önce 'fit' fonksiyonunu çalıştırın.")
        
        # NumPy'nin broadcasting özelliğini kullanarak tüm test ve train
        # örnekleri arasındaki farkı tek bir işlemde hesaplıyoruz.
        diffs = X[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
        
        if self.distance_metric == 'l1':
            # TODO: L1 mesafe hesaplama (Manhattan)
            distances = np.sum(np.abs(diffs), axis=2)
        elif self.distance_metric == 'l2':
            # TODO: L2 mesafe hesaplama (Euclidean)
            distances = np.sqrt(np.sum(diffs**2, axis=2))
        else:
            raise ValueError(f"Bilinmeyen mesafe metriği: {self.distance_metric}")
            
        return distances
    
    def predict(self, X):
        """
        Test örnekleri için tahmin yapar
        
        Parameters:
        -----------
        X : numpy array, shape (n_test, n_features)
            Test örnekleri
            
        Returns:
        --------
        predictions : numpy array, shape (n_test,)
            Tahmin edilen etiketler
        """
        if self.X_train is None:
            raise RuntimeError("Model 'fit' edilmedi. Lütfen önce 'fit' fonksiyonunu çalıştırın.")
        
        # 1. compute_distances() ile mesafeleri hesaplayın
        distances = self.compute_distances(X)
        
        n_test = X.shape[0]
        predictions = np.zeros(n_test)
        
        for i in range(n_test):
            # 2. Her test örneği için k en yakın komşuyu bulun
            # np.argsort mesafeleri küçükten büyüğe sıralar ve indisleri döndürür
            closest_k_indices = np.argsort(distances[i, :])[:self.k]
            k_nearest_labels = self.y_train[closest_k_indices]
            
            # 3. Majority voting ile sınıf tahmini yapın
            # Counter en sık geçen etiketi bulur
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions[i] = most_common[0][0]

        # 4. Tahminleri döndürün
        return predictions
    
    def score(self, X, y):
        """
        Model accuracy'sini hesaplar
        
        Parameters:
        -----------
        X : numpy array, shape (n_test, n_features)
            Test özellikleri
        y : numpy array, shape (n_test,)
            Gerçek etiketler
            
        Returns:
        --------
        accuracy : float
            Doğruluk skoru (0-1 arası)
        """
        # 1. predict() ile tahmin yapın
        y_pred = self.predict(X)
        
        # 2. Doğru tahmin sayısını / toplam örnek sayısını hesaplayın
        accuracy = np.mean(y_pred == y)
        
        return accuracy