# src/main.py
import os
import shutil

# === Diretórios principais do projeto ===
BASE_DIR = r"C:\Users\Laercio\PycharmProjects\PythonProject"

DIRS_TO_CREATE = [
    os.path.join(BASE_DIR, "data", "raw"),
    os.path.join(BASE_DIR, "data", "processed"),
    os.path.join(BASE_DIR, "outputs", "models"),
    os.path.join(BASE_DIR, "outputs", "plots"),
    os.path.join(BASE_DIR, "outputs", "metrics"),
    os.path.join(BASE_DIR, "outputs", "reports"),
    os.path.join(BASE_DIR, "models"),
    #os.path.join(BASE_DIR, "notebooks"),
    os.path.join(BASE_DIR, "src"),
]

# === Cria todos os diretórios, se não existirem ===
for d in DIRS_TO_CREATE:
    os.makedirs(d, exist_ok=True)
    print(f"Diretório garantido: {d}")

# === Diretórios que queremos limpar antes de processar novamente ===
DIRS_TO_CLEAN = [
    os.path.join(BASE_DIR, "data", "processed"),
    os.path.join(BASE_DIR, "outputs", "models"),
    os.path.join(BASE_DIR, "outputs", "plots"),
    os.path.join(BASE_DIR, "outputs", "metrics"),
    os.path.join(BASE_DIR, "outputs", "reports"),
]

for d in DIRS_TO_CLEAN:
    if os.path.exists(d):
        print(f"Limpando diretório: {d}")
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    print(f"Diretório recriado: {d}")

print("\nEstrutura de diretórios preparada e limpa. Agora você pode rodar cada script separadamente.")
