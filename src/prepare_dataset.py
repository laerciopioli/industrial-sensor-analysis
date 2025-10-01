# src/prepare_dataset.py
import os, json
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\Laercio\PycharmProjects\PythonProject\data\raw"
OUT_DIR = r"C:\Users\Laercio\PycharmProjects\PythonProject\data\processed" # salva os arquivos processados

files = ["Dados_1.npy", "Dados_2.npy", "Dados_3.npy", "Dados_4.npy", "Dados_5.npy"]
sensor_arrays = []
sensor_shapes = []

for fn in files:
    path = os.path.join(DATA_DIR, fn)
    arr = np.load(path, allow_pickle=True)
    print(f"Carregado {fn} shape {arr.shape}")
    # Se última coluna for NaN em toda coluna, NÃO remover automaticamente para preservar alinhamento.
    # A estratégia aqui é substituir NaN pela média da coluna.
    if np.isnan(arr).any():
        col_means = np.nanmean(arr, axis=0)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(col_means, inds[1])
    sensor_arrays.append(arr)
    sensor_shapes.append(arr.shape[1])

# decidir remover sensores inertes: detecta colunas com variância ~0
use_indices = []
for i, arr in enumerate(sensor_arrays):
    if np.nanstd(arr) < 1e-6:
        print(f"Sensor {i+1} tem variância ~0 (valores quase constantes). Será descartado.")
    else:
        use_indices.append(i)

# Concatena apenas sensores úteis
selected = [sensor_arrays[i] for i in use_indices]
X = np.concatenate(selected, axis=1)
print("X shape after concat:", X.shape)

# Carregar classes
classes = np.load(os.path.join(DATA_DIR, "Classes.npy"), allow_pickle=True).flatten()

# Verifica alinhamento
assert X.shape[0] == classes.shape[0], "Número de amostras difere entre X e classes!"

# monta dataframe
df = pd.DataFrame(X)
df["Classe"] = classes
out_parquet = os.path.join(OUT_DIR, "dataset_consolidado.parquet")
df.to_parquet(out_parquet, index=False)
print("Dataset salvo:", out_parquet)

# salva metadados (tamanhos por sensor e sensores usados)
meta = {"original_files": files, "sensor_shapes": sensor_shapes, "use_indices": use_indices}
with open(os.path.join(OUT_DIR, "meta_sensors.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("Metadados salvos.")
