import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

# Imports locais
from features import extract_features_dataframe

# === Diretórios padronizados ===
BASE_DIR = r"C:\Users\Laercio\PycharmProjects\PythonProject"
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# === Caminhos ===
dataset_path = os.path.join(DATA_PROCESSED, "features_dataset.parquet")
meta_path = os.path.join(DATA_PROCESSED, "meta_sensors.json")

# === Carrega dataset de features ===
print(" Lendo dataset de features...")
df = pd.read_parquet(dataset_path)
print(f"Dataset carregado com shape: {df.shape}")

# === Separa X e y ===
Xf = df.drop(columns=["Classe"])
y = df["Classe"]

# === Trata missing e normaliza ===
Xf = Xf.fillna(0.0)
scaler = StandardScaler()
Xs = scaler.fit_transform(Xf)

# === Treina modelo com validação cruzada ===
print(" Treinando Random Forest com validação cruzada...")
clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

y_pred = cross_val_predict(clf, Xs, y, cv=cv, n_jobs=-1)

print("\n Relatório de classificação:")
print(classification_report(y, y_pred))
print(f"F1 macro: {f1_score(y, y_pred, average='macro'):.4f}")
print("\nMatriz de confusão:")
print(confusion_matrix(y, y_pred))

# === Treino final ===
print("\n Treinando modelo final em todo o dataset...")
clf.fit(Xs, y)

# === Salva modelo ===
os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, "rf_baseline.joblib")
joblib.dump({"model": clf, "scaler": scaler}, model_path)

print(f"\n Modelo salvo com sucesso em: {model_path}")
