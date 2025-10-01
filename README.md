# 🔧 Projeto de Manutenção Preditiva com Aprendizado de Máquina

Este projeto apresenta o desenvolvimento completo de uma solução de **Inteligência Artificial aplicada à manutenção preditiva industrial**, desde o pré-processamento dos dados de sensores até a disponibilização do modelo de predição treinado.  
O sistema foi projetado para identificar padrões de comportamento nos sensores e prever condições de falha em máquinas industriais, contribuindo para a redução de paradas não planejadas e para o aumento da eficiência operacional.

---

## 📘 Sumário

1. [Visão Geral](#-visão-geral)
2. [Arquitetura do Projeto](#-arquitetura-do-projeto)
3. [Pipeline de Processamento](#-pipeline-de-processamento)
4. [Modelagem e Treinamento](#-modelagem-e-treinamento)
5. [Estrutura de Diretórios](#-estrutura-de-diretórios)
6. [Execução do Projeto](#-execução-do-projeto)
7. [Dependências e Ambiente](#-dependências-e-ambiente)
8. [Dados e Modelos](#-dados-e-modelos)
9. [Resultados](#-resultados)
10. [Autores e Créditos](#-autores-e-créditos)

---

## 🚀 Visão Geral

O objetivo deste projeto é **construir um pipeline de aprendizado de máquina** capaz de:

- Processar dados de múltiplos sensores industriais;
- Extrair **características relevantes (features)** de séries temporais;
- Treinar e avaliar um modelo supervisionado para **classificação do estado da máquina** (normal, falha, etc.);
- Disponibilizar uma solução modular, interpretável e reprodutível.

O projeto integra **conceitos teóricos e práticos** de Engenharia de Software, Ciência de Dados e Aprendizado de Máquina aplicados ao contexto industrial.

---

## 🧩 Arquitetura do Projeto

A solução foi projetada com base em um **pipeline de machine learning modular**, composto por cinco etapas principais:

1. **Pré-processamento** – Limpeza e normalização dos sinais de sensores, tratamento de valores ausentes e remoção de sensores redundantes;
2. **Engenharia de Features** – Extração de métricas estatísticas e representações numéricas dos sinais;
3. **Treinamento do Modelo** – Aplicação do algoritmo **Random Forest**, escolhido por sua interpretabilidade e robustez;
4. **Avaliação de Desempenho** – Análise de métricas de erro, acurácia e correlação;
5. **Disponibilização da Solução** – Organização modular dos scripts e dos resultados em repositório versionado.

---

## ⚙️ Pipeline de Processamento

A estrutura abaixo representa o fluxo lógico do pipeline desenvolvido:

[ Dados Brutos ]
↓
[ Pré-processamento ]
↓
[ Extração de Features ]
↓
[ Treinamento do Modelo ]
↓
[ Avaliação e Visualização de Resultados ]


Cada etapa foi implementada em um módulo independente, garantindo clareza, modularidade e possibilidade de reutilização.

---

## 🤖 Modelagem e Treinamento

- **Algoritmo escolhido:** Random Forest Classifier  
- **Motivos da escolha:**
  - Capacidade de lidar com grande número de features;
  - Robustez frente a ruídos e dados desbalanceados;
  - Possibilidade de interpretação das features mais importantes;
  - Boa performance com baixo custo computacional.

Durante o treinamento, foram aplicadas técnicas de:
- **Validação cruzada (cross-validation)**;
- **Análise de overfitting**;
- **Interpretação de importância das features**.

As métricas de avaliação incluíram:
- **RMSE (Root Mean Square Error)**  
- **MAE (Mean Absolute Error)**  
- **Correlação entre valores previstos e reais**

---

## 📂 Estrutura de Diretórios

A organização do projeto reflete a divisão lógica das etapas do pipeline:
PythonProject/
│
├── data/
│ - raw/ → Dados brutos dos sensores (.npy)
│ - processed/ → Dados consolidados e tratados
│
├── models/
│ - rf_model.pkl → Modelo Random Forest treinado
│ - metrics/ → Relatórios e métricas de avaliação
│
├── src/
│ - preprocessing.py → Funções de limpeza e normalização dos dados
│ - features.py → Extração de características
│ - train_model.py → Treinamento do modelo
│ - predict.py → Geração de previsões
│ - utils.py → Funções auxiliares
│
├── notebooks/
│ - ExploratoryAnalysis.ipynb → Análises exploratórias dos dados
│ - ModelEvaluation.ipynb → Avaliação das métricas e visualização
│
├── outputs/
│ - plots/ → Gráficos gerados durante o treinamento
│ - reports/ → Relatórios de execução
│
- requirements.txt → Dependências do projeto
- main.py → Script principal de execução do pipeline
- README.md → Documento de descrição e instruções


---

## ▶️ Execução do Projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/laerciopioli/industrial-sensor-analysis.git
cd industrial-sensor-analysis

2. Criar e ativar um ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows

3. Instalar as dependências
pip install -r requirements.txt

4. Executar o pipeline completo
python main.py


☁️ Dados e Modelos
Os arquivos de grande porte, como datasets processados e modelos treinados, foram disponibilizados em Google Drive, devido à limitação de tamanho do GitHub (máximo de 100 MB por arquivo).

📦 Arquivos disponíveis no Drive:
Dados de entrada (Dados_1.npy, Dados_2.npy, ..., Classes.npy);
Dataset consolidado (dataset_consolidado.parquet);
Dataset de features (features_dataset.parquet);
Modelo treinado (rf_baseline.joblib).

🔗 Link de acesso aos dados e modelos: https://drive.google.com/drive/folders/1Oq0BmgjEEEDd_vegdJYFn7WtDLlo7T3F?usp=sharing
👉 Acessar Google Drive

📊 Resultados

Durante os testes, o modelo apresentou:
Boa capacidade de generalização entre classes distintas;
Alta estabilidade entre diferentes execuções (baixa variância);
Relevância consistente das features relacionadas aos sensores de maior variação.
As visualizações disponíveis em outputs/plots/ incluem:
Matriz de confusão;
Curvas de aprendizado;
Gráficos de importância das features.
Esses resultados comprovam a eficiência do pipeline proposto na predição de estados de máquina com base em sinais sensoriais industriais.

👨‍💻 Autores e Créditos

Autor: Laércio Pioli
Instituição: FIESC / SENAI
Área: Inteligência Artificial Aplicada à Indústria

📧 Contato: laerciopiolijr@gmail.com
