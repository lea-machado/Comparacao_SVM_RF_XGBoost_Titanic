# 🚢 Comparação de Modelos de ML para Previsão de Sobrevivência no Titanic

Este projeto utiliza o famoso dataset "Titanic" (disponível na biblioteca Seaborn) para construir e avaliar três modelos de Machine Learning (SVM, Random Forest e XGBoost) com o objetivo de prever a sobrevivência dos passageiros.

O foco principal é a criação de um pipeline de pré-processamento robusto e a otimização de hiperparâmetros usando `RandomizedSearchCV` para encontrar o modelo de melhor desempenho.

**Notebook Principal:** `SVM,_RF_e_XGBoost_dataset_Titanic.ipynb`

---

## ⚙️ Metodologia e Pré-processamento

O pré-processamento foi uma etapa crucial deste projeto, gerenciado inteiramente através de `Pipelines` e `ColumnTransformer` do Scikit-learn para evitar vazamento de dados (data leakage).

As principais etapas foram:

1.  **Tratamento de Dados Faltantes:**
    * **`deck`**: A coluna foi **removida**, pois 77% dos seus dados estavam ausentes (688 de 891).
    * **`age`**: Os 177 valores ausentes foram preenchidos usando a **mediana**, uma estratégia robusta contra os valores extremos (outliers) de idade no dataset.
    * **`embarked`**: Os 2 valores ausentes foram preenchidos usando a **moda** (valor mais frequente).

2.  **Engenharia de Features (Encoding & Scaling):**
    * **Features Categóricas** (`sex`, `embarked`, `pclass`): Foram transformadas em colunas numéricas usando `OneHotEncoder`.
    * **Features Numéricas** (`age`, `fare`, `sibsp`, `parch`): Foram padronizadas (colocadas na mesma escala) usando `StandardScaler`.

3.  **Otimização de Modelos:**
    * Os três modelos (`SVC`, `RandomForestClassifier`, `XGBClassifier`) tiveram seus hiperparâmetros otimizados via `RandomizedSearchCV` (com 50 iterações).
    * A métrica principal para otimização foi a **ROC-AUC**.
    * A validação foi feita usando `StratifiedKFold` (5 splits) para garantir que a proporção de sobreviventes/não-sobreviventes fosse mantida em cada "fold".

---

## 📊 Resultados e Análise

Após o treinamento e a otimização, os modelos foram avaliados no conjunto de teste (20% dos dados).

### 1. Modelo Vencedor: XGBoost

O **XGBoost** apresentou o melhor desempenho geral em todas as métricas avaliadas.

A tabela abaixo resume a performance final no conjunto de teste:

| Modelo | ROC-AUC | Acurácia | F1-Score | VP | FN | FP | VN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | **0.8861** | **0.8156** | **0.7591** | **52** | **22** | **11** | **94** |
| Random Forest | 0.8825 | 0.7933 | 0.7218 | 48 | 26 | 11 | 94 |
| SVM | 0.8784 | 0.7877 | 0.7286 | 51 | 23 | 15 | 90 |

As curvas ROC e Precision-Recall também confirmam visualmente o desempenho superior do XGBoost.

### 2. Análise de Overfitting

**Nenhum modelo apresentou sinais de overfitting.**

A tabela abaixo compara o score ROC-AUC da Validação Cruzada (treino) com o score final no conjunto de Teste.

| Modelo | Score Treino (CV) | Score Teste | Diferença (Teste - Treino) |
| :--- | :--- | :--- | :--- |
| XGBoost | 0.8476 | 0.8861 | **+0.0385** |
| Random Forest | 0.8576 | 0.8825 | **+0.0249** |
| SVM | 0.8588 | 0.8784 | **+0.0196** |

Todos os modelos apresentaram um desempenho **melhor** nos dados de teste do que na média dos dados de treino, indicando uma excelente capacidade de generalização.

---

## 🚀 Instruções de Reexecução

Para reexecutar este projeto e reproduzir os resultados:

1.  **Clone este repositório:**
    ```bash
    git clone https://github.com/lea-machado/Comparacao_SVM_RF_XGBoost_Titanic.git
    cd Comparacao_SVM_RF_XGBoost_Titanic
    ```

2.  **Crie um ambiente virtual (Recomendado):**
    ```bash
    python -m venv venv
    
    # No Linux/Mac
    source venv/bin/activate
   
    # No Windows (PowerShell/CMD)
    .\venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    O projeto utiliza bibliotecas padrão de Data Science.
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyterlab
    ```

4.  **Execute o Notebook:**
    Abra o arquivo `SVM,_RF_e_XGBoost_dataset_Titanic.ipynb` usando Jupyter Lab ou Google Colab e execute as células em ordem.

    *Não é necessário baixar nenhum arquivo CSV*, pois o dataset é carregado diretamente da biblioteca Seaborn (`sns.load_dataset("titanic")`).
