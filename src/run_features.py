import os
import pandas as pd
import json
from features import extract_features_dataframe

# === Diretórios ===
BASE_DIR = r"C:\Users\Laercio\PycharmProjects\PythonProject"
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# === Caminhos dos arquivos ===
DATASET_PATH = os.path.join(DATA_PROCESSED_DIR, "dataset_consolidado.parquet")
META_PATH = os.path.join(DATA_PROCESSED_DIR, "meta_sensors.json")
FEATURES_OUT_PATH = os.path.join(DATA_PROCESSED_DIR, "features_dataset.parquet")

# === Criar diretório de saída, se não existir ===
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

print("=== Iniciando extração de features ===")
print(f"Lendo dataset consolidado de: {DATASET_PATH}")

# === Carregar dataset consolidado ===
df = pd.read_parquet(DATASET_PATH)
print(f" Dataset lido com shape: {df.shape}")

# === Garantir que as colunas numéricas estão no formato correto ===
for col in df.columns[:-1]:  # ignora a última coluna (Classe)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === Verificar e tratar NaN ===
df = df.fillna(0.0)
print(" Dados convertidos e limpos")

# === Extração de features ===
print("🔍 Extraindo features...")
Xf, y = extract_features_dataframe(df, META_PATH, sr=10000.0, max_rows=None)

print(f" Features extraídas com shape: {Xf.shape}")
print(f" Número de classes detectadas: {y.nunique()}")

# === Adicionar coluna de classe ===
Xf["Classe"] = y.values

# === Salvar dataset de features ===
Xf.to_parquet(FEATURES_OUT_PATH, index=False)
print(f" Dataset de features salvo em: {FEATURES_OUT_PATH}")
