# 📊 Análise de Fatores de Cancelamento em Academias (Estudo de Caso)

---

Este repositório contém um estudo de caso detalhado de **análise estatística** focado na identificação dos principais fatores que levam ao cancelamento de clientes em academias. O objetivo é fornecer insights acionáveis para estratégias de retenção.

### **Visão Geral do Projeto**

O projeto explora um conjunto de dados abrangente para entender o perfil de clientes que cancelam, as variáveis mais correlacionadas com a decisão de cancelar e a construção de um modelo preditivo para identificar clientes em risco. A análise segue um fluxo completo, desde a exploração inicial dos dados até a geração de recomendações de negócio.

### **Fonte dos Dados**

A base de dados utilizada neste estudo de caso foi gentilmente cedida como parte de um **treinamento da Hashtag Treinamentos**, focando no setor de academias. É um dataset sintético, mas representativo, projetado para demonstrar técnicas de análise de dados.

### **Metodologia e Análises Realizadas**

1.  **Carregamento e Pré-processamento de Dados**: Tratamento de valores ausentes, padronização de colunas e preparação de dados para modelagem.
2.  **Análise Exploratória (EDA)**: Estatísticas descritivas e visualizações para entender a distribuição das variáveis e a taxa geral de cancelamento.
3.  **Análise de Associação Categórica (Qui-Quadrado)**: Investigação da relação entre variáveis categóricas (como sexo, tipo de assinatura, duração do contrato) e a propensão ao cancelamento.
4.  **Construção de Modelo Preditivo (Regressão Logística)**: Desenvolvimento de um modelo para prever a probabilidade de um cliente cancelar.
5.  **Análise de Fatores de Risco**: Identificação das variáveis mais impactantes no risco de cancelamento, com base nos coeficientes do modelo.
6.  **Segmentação de Clientes por Risco**: Categorização de clientes em grupos de alto, médio e baixo risco de cancelamento.
7.  **Insights e Recomendações de Negócio**: Tradução dos achados estatísticos em recomendações estratégicas para a retenção de clientes.

### **Tecnologias e Bibliotecas**

* **Python**: Linguagem de programação principal.
* **Pandas**: Manipulação e análise de dados.
* **NumPy**: Operações numéricas.
* **Matplotlib** & **Seaborn**: Geração de visualizações e gráficos estatísticos.
* **SciPy**: Testes estatísticos (Qui-Quadrado).
* **Scikit-learn**: Construção e avaliação do modelo de Regressão Logística.
* **Statsmodels**: Análise estatística (embora não diretamente utilizado na versão final, é uma biblioteca comum neste contexto).

### **Como Executar o Projeto**

1.  **Clone o Repositório**:

    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```
2.  **Abra no Google Colab**: Faça o upload do arquivo `.ipynb` (ou cole o código em um novo notebook Colab).
3.  **Carregue o Dataset**: Certifique-se de que o arquivo `cancelamentos.csv` está no ambiente do Colab (você pode fazer upload manual para a sessão ou configurá-lo no Google Drive).
4.  **Execute as Células**: Execute as células do notebook sequencialmente.

### **Principais Insights**

* Identificação de variáveis-chave que influenciam o cancelamento (ex: **número de ligações ao call center**, **duração do contrato**).
* Visualização clara da distribuição das variáveis e das taxas de cancelamento por categoria.
* Um modelo preditivo capaz de classificar clientes com uma acurácia razoável, permitindo ações proativas de retenção.
* Segmentação de clientes em grupos de risco para campanhas direcionadas.


---

<br>

---

# 📊 Customer Churn Analysis in Gyms (Case Study)

---

This repository contains a detailed **statistical analysis** case study focused on identifying the main factors that lead to customer churn in gyms. The objective is to provide actionable insights for retention strategies.

### **Project Overview**

The project explores a comprehensive dataset to understand the profile of churning customers, the variables most correlated with the decision to cancel, and the construction of a predictive model to identify at-risk customers. The analysis follows a complete flow, from initial data exploration to generating business recommendations.

### **Data Source**

The dataset used in this case study was kindly provided as part of a **Hashtag Treinamentos training**, focusing on the gym sector. It is a synthetic but representative dataset, designed to demonstrate data analysis techniques.

### **Methodology and Analyses Performed**

1.  **Data Loading and Preprocessing**: Handling missing values, standardizing columns, and preparing data for modeling.
2.  **Exploratory Data Analysis (EDA)**: Descriptive statistics and visualizations to understand variable distributions and the overall churn rate.
3.  **Categorical Association Analysis (Chi-Square)**: Investigating the relationship between categorical variables (such as gender, subscription type, contract duration) and the propensity to churn.
4.  **Predictive Model Building (Logistic Regression)**: Developing a model to predict the probability of a customer churning.
5.  **Risk Factor Analysis**: Identifying the most impactful variables on churn risk, based on the model's coefficients.
6.  **Customer Segmentation by Risk**: Categorizing customers into high, medium, and low churn risk groups.
7.  **Business Insights and Recommendations**: Translating statistical findings into strategic recommendations for customer retention.

### **Technologies and Libraries**

* **Python**: Primary programming language.
* **Pandas**: Data manipulation and analysis.
* **NumPy**: Numerical operations.
* **Matplotlib** & **Seaborn**: Statistical visualization and graphing.
* **SciPy**: Statistical tests (Chi-Square).
* **Scikit-learn**: Building and evaluating the Logistic Regression model.
* **Statsmodels**: Statistical analysis (though not directly used in the final version, it's a common library in this context).

### **How to Run the Project**

1.  **Clone the Repository**:

    ```bash
    git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
    cd your-repository
    ```
2.  **Open in Google Colab**: Upload the `.ipynb` file (or paste the code into a new Colab notebook).
3.  **Load the Dataset**: Ensure the `cancelamentos.csv` file is in the Colab environment (you can manually upload it to the session or configure it from Google Drive).
4.  **Execute Cells**: Run the notebook cells sequentially.

### **Key Insights**

* Identification of key variables influencing churn (e.g., **number of call center interactions**, **contract duration**).
* Clear visualization of variable distributions and churn rates by category.
* A predictive model capable of classifying customers with reasonable accuracy, enabling proactive retention actions.
* Segmentation of customers into risk groups for targeted campaigns.

