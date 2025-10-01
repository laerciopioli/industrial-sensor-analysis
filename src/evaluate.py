import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

# === Diretórios base padronizados ===
BASE_DIR = r"C:\Users\Laercio\PycharmProjects\PythonProject"
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# === Garantir que diretórios existam ===
os.makedirs(PLOTS_DIR, exist_ok=True)

# === Caminhos ===
MODEL_PATH = os.path.join(MODELS_DIR, "rf_baseline.joblib")
FEATURES_PATH = os.path.join(DATA_PROCESSED, "features_dataset.parquet")

# === Carregar dataset de features ===
print(" Carregando dataset de features...")
df = pd.read_parquet(FEATURES_PATH)
X = df.drop(columns=["Classe"])
y = df["Classe"]
print(f" Dataset carregado com shape: {df.shape}")

# === Carregar modelo e scaler ===
print("🔍 Carregando modelo treinado...")
loaded = load(MODEL_PATH)
model = loaded["model"]
scaler = loaded["scaler"]
print(f" Modelo carregado com sucesso de: {MODEL_PATH}")

# === Normalizar X antes da previsão ===
X_scaled = scaler.transform(X)

# === Validação cruzada ===
print("\n Executando validação cruzada...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv, n_jobs=-1)

# === Relatório de métricas ===
print("\n Relatório de classificação (cross-validation):")
print(classification_report(y, y_pred_cv))
f1_macro = f1_score(y, y_pred_cv, average="macro")
print(f" F1 macro: {f1_macro:.4f}")

# === Matriz de confusão ===
cm = confusion_matrix(y, y_pred_cv, labels=model.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Classe Predita")
plt.ylabel("Classe Real")
plt.title("Matriz de Confusão - Random Forest (Cross-Validation)")
confusion_path = os.path.join(PLOTS_DIR, "confusion_matrix_cv.png")
plt.savefig(confusion_path, bbox_inches="tight")
plt.close()
print(f" Matriz de confusão salva em: {confusion_path}")

# === Importância das features ===
importances = model.feature_importances_
feat_names = X.columns
feat_imp_df = (
    pd.DataFrame({"feature": feat_names, "importance": importances})
    .sort_values(by="importance", ascending=False)
)

plt.figure(figsize=(12,6))
sns.barplot(data=feat_imp_df.head(20), x="importance", y="feature", palette="viridis")
plt.title("Top 20 Features - Random Forest")
plt.xlabel("Importância")
plt.ylabel("Feature")
feat_imp_path = os.path.join(PLOTS_DIR, "feature_importance_cv.png")
plt.savefig(feat_imp_path, bbox_inches="tight")
plt.close()
print(f" Gráfico de importância de features salvo em: {feat_imp_path}")

print("\n Avaliação concluída com sucesso!")
