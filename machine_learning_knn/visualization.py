# Dosya Adı: visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='results/confusion_matrix.png'):
    """
    Confusion matrix görselleştirmesi
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.savefig(save_path)
    plt.show()

def plot_sample_predictions(X_test, y_test, y_pred, n_samples=10,
                           save_path='results/sample_predictions.png'):
    """
    Örnek tahminleri görselleştirir
    """
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 4))

    for i, idx in enumerate(indices):
        image = X_test[idx].reshape(8, 8)
        true_label = y_test[idx]
        pred_label = y_pred[idx]

        axes[i].imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f"Gerçek: {true_label}\nTahmin: {pred_label}", color=color)
        axes[i].axis('off')

    plt.suptitle('Örnek Tahminler')
    plt.savefig(save_path)
    plt.show()

def plot_k_analysis(k_values, accuracies, save_path='results/k_value_analysis.png'):
    """
    K değeri analizi grafiği
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='--')

    best_k_idx = np.argmax(accuracies)
    best_k = k_values[best_k_idx]
    best_acc = accuracies[best_k_idx]
    plt.scatter(best_k, best_acc, color='red', zorder=5, label=f'En İyi K: {best_k} (Acc: {best_acc:.4f})')

    plt.xlabel('K Değeri')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.title('K Değerinin Doğruluğa Etkisi (L2 Mesafesi)')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def plot_distance_comparison(k_values, l1_accuracies, l2_accuracies,
                             save_path='results/distance_comparison.png'):
    """
    L1 ve L2 mesafe metriklerini karşılaştırır
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, l1_accuracies, marker='o', linestyle='--', label='L1 (Manhattan)')
    plt.plot(k_values, l2_accuracies, marker='s', linestyle='-', label='L2 (Euclidean)')

    plt.legend()
    plt.xlabel('K Değeri')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.title('Mesafe Metriklerinin Karşılaştırılması')
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

def create_comparison_table(k_values, l1_accuracies, l2_accuracies,
                            save_path='results/comparison_table.png'):
    """
    Karşılaştırma tablosu oluşturur
    """
    data = []
    l1_accuracies = np.array(l1_accuracies)
    l2_accuracies = np.array(l2_accuracies)
    differences = l2_accuracies - l1_accuracies

    for i, k in enumerate(k_values):
        data.append([k, f"{l1_accuracies[i]:.4f}", f"{l2_accuracies[i]:.4f}", f"{differences[i]:.4f}"])

    columns = ('K Değeri', 'L1 Accuracy', 'L2 Accuracy', 'Fark (L2-L1)')

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.title('', y=0.7)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.show()