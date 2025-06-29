import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Tekrar sayısı
n_repeats = 10

# Veri yükleme
df = pd.read_excel("D:/OneDrive - Bartın Üniversitesi/MAKALE/2025/iris_deep_learning/image_results.xlsx")
df["person_id"] = df["File"].apply(lambda x: int(x.split("_")[0]))
labels = df["person_id"].values
features = df.drop(columns=["File", "person_id"]).values.astype(np.float32)

# Normalizasyon (test verisine sızma yok çünkü split sonrası fit ediliyor)
scaler = MinMaxScaler()

# Model tanımı
model = LogisticRegression(C=10, max_iter=2000, solver='lbfgs')

# Sonuçları sakla
all_accuracies = []

# Tekrarlı eğitim
for repeat in range(n_repeats):
    print(f"\n==== Repeat {repeat+1} ====")

    # Eğitim/test ayırma
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=repeat
    )

    # Normalizasyon fit sadece eğitim verisine yapılır
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # PCA da yalnızca eğitim verisiyle fitlenir
    pca = PCA(n_components=0.99, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Model eğitimi ve test
    start = time.time()
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    all_accuracies.append(acc)
    print(f"Accuracy: {acc:.4f} ({time.time() - start:.2f} sec)")

# Sonuçları DataFrame'e aktar
results_df = pd.DataFrame({"Repeat": list(range(1, n_repeats+1)), "Accuracy": all_accuracies})
mean_acc = results_df["Accuracy"].mean()
std_acc = results_df["Accuracy"].std()

# Box plot
plt.figure(figsize=(8, 5), dpi=300)
plt.boxplot(all_accuracies, labels=["Logistic Regression"])
plt.title("Accuracy Distribution over Repeats (Proper PCA)")
plt.ylabel("Accuracy")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("logreg_boxplot_corrected.png", dpi=300)

# Excel çıktısı
with pd.ExcelWriter("logreg_results_corrected.xlsx") as writer:
    results_df.to_excel(writer, sheet_name="AccuracyPerRun", index=False)
    pd.DataFrame({
        "Model": ["Logistic Regression"],
        "Mean": [mean_acc],
        "Std": [std_acc],
        "Mean ± Std": [f"{mean_acc:.4f} ± {std_acc:.4f}"]
    }).to_excel(writer, sheet_name="SummaryStats", index=False)

print("\nTüm çıktılar başarıyla kaydedildi:")
print("- logreg_results_corrected.xlsx")
print("- logreg_boxplot_corrected.png")
