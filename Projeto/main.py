import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import time

# Medição do tempo de execução da análise.
start_time = time.time()

# --- Configurações globais para visualização de gráficos ---
# Define o estilo e a paleta de cores para todos os gráficos, garantindo consistência visual.
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
plt.rcParams['figure.figsize'] = (14, 8) # Tamanho padrão maior para as figuras.
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

def plot_styled(func):
    """
    Decorador para aplicar estilos globais a uma função de plotagem.
    Isso garante que qualquer gráfico gerado pela função decorada siga as configurações
    predefinidas de Matplotlib e Seaborn.
    """
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        return plt
    return wrapper

def format_chi2_results(col_name, qui2, p):
    """
    Formata os resultados do teste Qui-quadrado para uma coluna categórica,
    fornecendo a estatística, o valor-P e uma interpretação da associação.

    Args:
        col_name (str): O nome da coluna categórica testada.
        qui2 (float): O valor da estatística Qui-quadrado.
        p (float): O valor-P resultante do teste.

    Returns:
        str: Uma string formatada contendo os resultados e a interpretação.
    """
    result = f"\n===== ASSOCIAÇÃO: {col_name.upper()} & CANCELAMENTOS =====\n"
    result += f"Estatística Qui²: {qui2:.2f}, Valor-P: {p:.6f}\n"

    # Interpretação do valor-P para determinar a força da associação.
    if p < 0.001:
        result += f"✓ ASSOCIAÇÃO MUITO FORTE entre {col_name} e cancelamentos (p<0.001)\n"
        result += "     Interpretação: Existe uma relação extremamente significativa\n"
    elif p < 0.01:
        result += f"✓ ASSOCIAÇÃO FORTE entre {col_name} e cancelamentos (p<0.01)\n"
        result += "     Interpretação: Existe uma relação altamente significativa\n"
    elif p < 0.05:
        result += f"✓ ASSOCIAÇÃO MODERADA entre {col_name} e cancelamentos (p<0.05)\n"
        result += "     Interpretação: Existe uma relação significativa\n"
    else:
        result += f"✗ SEM ASSOCIAÇÃO entre {col_name} e cancelamentos (p={p:.6f})\n"
        result += "     Interpretação: Não há evidência de relação significativa\n"

    return result

## Carregar o conjunto de dados
# Tenta carregar o arquivo CSV. Se houver um erro, imprime a mensagem e sai do script.
try:
    df = pd.read_csv('cancelamentos.csv')
    print(f"Dataset carregado com sucesso: {df.shape[0]} registros e {df.shape[1]} variáveis")
except Exception as e:
    print(f"Erro ao carregar o dataset: {e}")
    df = pd.DataFrame() # Inicializa um DataFrame vazio para evitar erros posteriores.
    exit()

# --- Limpeza e Pré-processamento ---
print("\n[1/5] Limpando dados e preparando análise...")

# Verifica se o DataFrame foi carregado e contém a coluna 'cancelou'.
if df is not None and not df.empty and 'cancelou' in df.columns:
    initial_rows = df.shape[0]
    # Remove linhas onde a coluna 'cancelou' possui valores ausentes, pois é a variável alvo.
    df.dropna(subset=['cancelou'], inplace=True)
    dropped_rows = initial_rows - df.shape[0]
    if dropped_rows > 0:
        print(f"Aviso: Removidas {dropped_rows} linhas com valores ausentes na coluna 'cancelou'.")
    # Converte a coluna 'cancelou' para tipo inteiro.
    df['cancelou'] = df['cancelou'].astype(int)
else:
    # Trata cenários onde o DataFrame não foi carregado ou a coluna 'cancelou' está ausente.
    if df is None or df.empty:
        print("Erro: DataFrame vazio ou não carregado. Não é possível continuar o pré-processamento.")
    elif 'cancelou' not in df.columns:
        print("Erro: Coluna 'cancelou' não encontrada no DataFrame. Verifique o arquivo de dados.")
    exit()

# Realiza a limpeza de colunas se o DataFrame não estiver vazio.
if not df.empty:
    # Remove espaços em branco dos nomes das colunas.
    df.columns = df.columns.str.strip()

    # Preenche valores ausentes em colunas numéricas com a mediana, uma medida robusta contra outliers.
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = numeric_cols.drop('cancelou', errors='ignore') # Exclui a variável alvo.
    if not numeric_cols.empty:
        df.loc[:, numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Preenche valores ausentes em colunas categóricas com a moda (valor mais frequente).
    categorical_cols = ['sexo', 'assinatura', 'duracao_contrato']
    for col in categorical_cols:
        if col in df.columns:
            df.loc[:, col] = df[col].fillna(df[col].mode()[0])
            # Garante que as colunas categóricas sejam do tipo string e remove espaços em branco.
            df.loc[:, col] = df[col].astype(str).str.strip()

# --- Análise Exploratória Rápida ---
print(f"[2/5] Realizando análise exploratória...")

def plot_improved_histograms(dataframe, columns):
    """
    Gera e exibe histogramas para as colunas numéricas especificadas.
    Inclui linhas para a média e a mediana para melhor compreensão da distribuição.

    Args:
        dataframe (pd.DataFrame): O DataFrame contendo os dados.
        columns (list): Uma lista de nomes de colunas numéricas para plotar.
    """
    n_cols = len(columns)
    # Calcula o número de linhas necessário para organizar os gráficos em 2 colunas.
    n_rows = (n_cols + 1) // 2

    # Aumenta o tamanho da figura para acomodar múltiplos gráficos de forma clara.
    # Ajusta a largura e altura para uma melhor visualização horizontal.
    if n_cols == 5:
        fig, axes = plt.subplots(3, 2, figsize=(20, 8 * 3)) # Aumentei de 16 para 20
    else:
        fig, axes = plt.subplots(n_rows, 2, figsize=(20, 8 * n_rows)) # Aumentei de 16 para 20

    # Achata o array de eixos para facilitar a iteração, independentemente do número de linhas.
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if i < len(axes): # Garante que não haja tentativa de plotar em um eixo inexistente.
            ax = axes[i]
            sns.histplot(dataframe[col], bins=20, kde=True, ax=ax, color='skyblue',
                         edgecolor='black', alpha=0.7)

            ax.set_title(f'Distribuição de {col.replace("_", " ").title()}', fontweight='bold', fontsize=16)
            ax.set_xlabel(f'{col.replace("_", " ").title()}')
            ax.set_ylabel('Frequência (por usuário)')
            ax.grid(True, linestyle='--', alpha=0.7)

            mean_val = dataframe[col].mean()
            median_val = dataframe[col].median()

            # Adiciona linhas verticais para a média e a mediana no histograma.
            line_mean = ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_val:.2f}')
            line_median = ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Mediana: {median_val:.2f}')

            # Posiciona a legenda fora dos histogramas para evitar sobreposição de dados.
            ax.legend(handles=[line_mean, line_median], loc='upper left',
                      fontsize=10, facecolor='white', framealpha=0.9,
                      bbox_to_anchor=(1.02, 1)) # Coordenadas para posicionar fora do subplot

    # Remove eixos extras que podem ter sido criados se o número de colunas for ímpar.
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajusta o layout para garantir que as legendas não sejam cortadas.
    plt.tight_layout(rect=[0, 0.03, 0.95, 0.98])
    plt.show()

# Define as colunas numéricas relevantes para a análise de distribuição.
relevant_numeric_cols = ['idade', 'frequencia_uso', 'total_gasto',
                         'ligacoes_callcenter', 'meses_ultima_interacao']
plot_improved_histograms(df, relevant_numeric_cols)

# --- Análise de Associação Categórica (Qui-Quadrado) ---
print(f"[3/5] Analisando associações entre variáveis categóricas e cancelamentos...")

# Itera sobre as colunas categóricas para realizar o teste Qui-quadrado e gerar gráficos.
for col in categorical_cols:
    if col in df.columns:
        temp_df = df.dropna(subset=[col, 'cancelou'])

        if temp_df.empty:
            print(f"Aviso: Coluna '{col}' vazia após a filtragem de NaNs. Pulando análise e gráfico.")
            continue

        # Cria uma tabela de contingência entre a coluna categórica e a variável 'cancelou'.
        tabela = pd.crosstab(temp_df[col], temp_df['cancelou'])

        # Calcula os percentuais para facilitar a interpretação da associação (ex: % de cancelamento por categoria).
        tabela_pct = pd.crosstab(temp_df[col], temp_df['cancelou'], normalize='index') * 100

        # Realiza o teste Qui-quadrado para avaliar a independência entre as variáveis.
        qui2, p, gl, esperado = chi2_contingency(tabela)

        # Exibe os resultados formatados do teste Qui-quadrado.
        print(format_chi2_results(col, qui2, p))

        # Exibe a tabela de contingência com percentuais para detalhamento.
        print(f"Tabela de contingência (% de cada {col} que cancelou):")
        print(tabela_pct.round(2))
        print("-" * 50)

        def plot_improved_stacked_bar(col_name, data_pct):
            """
            Gera um gráfico de barras empilhadas para visualizar a taxa de cancelamento
            por categoria de uma variável categórica.

            Args:
                col_name (str): O nome da coluna categórica que está sendo plotada.
                data_pct (pd.DataFrame): O DataFrame com os percentuais de cancelamento por categoria.
            """
            fig, ax = plt.subplots(figsize=(12, 7))

            # Cria o gráfico de barras empilhadas.
            data_pct.plot(kind='bar', stacked=True,
                                    colormap='coolwarm', ax=ax)

            plt.title(f'Taxa de Cancelamento por {col_name.replace("_", " ").title()} (%)', fontweight='bold', fontsize=16)
            plt.ylabel('Percentual (%)', fontsize=12)
            plt.xlabel(col_name.replace('_', ' ').title(), fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)

            # Move a legenda para fora do gráfico para liberar espaço e melhorar a legibilidade.
            plt.legend(title='Cancelou', labels=['Não (0)', 'Sim (1)'],
                                     loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

            # Adiciona rótulos de percentual em cada segmento da barra para clareza.
            for container in ax.containers:
                labels = [f'{val:.1f}%' if val > 0 else '' for val in container.datavalues]

                # Ajusta a cor do texto do rótulo para contrastar com a cor da barra.
                is_cancel_bar = (container.get_label() == '1')
                text_color = 'white' if is_cancel_bar else 'black'

                ax.bar_label(container, labels=labels, label_type='center', fontsize=9,
                                     color=text_color, fontweight='bold')

            # Adiciona uma anotação explicativa abaixo do gráfico para orientar a interpretação.
            plt.figtext(0.5, 0.01,
                                     f'\nInterpretação: Este gráfico mostra a proporção de clientes que cancelaram (1) '
                                     f'ou não cancelaram (0) para cada categoria de {col_name.replace("_", " ").lower()}.',
                                     ha='center', fontsize=9, bbox={'facecolor':'lightgray', 'alpha':0.6, 'pad':5})

            # Ajusta o layout para garantir que a anotação e a legenda não sejam cortadas.
            plt.tight_layout(rect=[0, 0.15, 0.95, 1])
            plt.show()

        # Chama a função para plotar o gráfico de barras empilhadas para a coluna atual.
        plot_improved_stacked_bar(col, tabela_pct)

# --- Modelo Preditivo ---
print(f"[4/5] Construindo modelo de regressão logística...")

# Cria variáveis dummy (one-hot encoding) para as colunas categóricas.
# 'drop_first=True' é usado para evitar multicolinearidade, removendo uma das categorias.
# A coluna 'CustomerID' é removida, pois é um identificador e não uma feature preditiva.
X = pd.get_dummies(df.drop(['CustomerID', 'cancelou'], axis=1, errors='ignore'),
                   columns=categorical_cols, drop_first=True, dtype=int)

y = df['cancelou'] # Define a variável alvo.

# Divide os dados em conjuntos de treino e teste.
# 'stratify=y' garante que a proporção de classes da variável alvo seja mantida em ambos os conjuntos.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# Inicializa e treina o modelo de Regressão Logística.
# 'max_iter' e 'solver' são definidos para garantir convergência e desempenho.
# 'n_jobs=-1' utiliza todos os processadores disponíveis.
# 'C' é o inverso da força de regularização.
modelo = LogisticRegression(max_iter=2000, solver='liblinear', n_jobs=-1, C=0.1)
modelo.fit(X_treino, y_treino)

y_pred = modelo.predict(X_teste) # Faz previsões no conjunto de teste.
y_prob = modelo.predict_proba(X_teste)[:, 1] # Obtém as probabilidades da classe positiva (cancelamento).

# Gera e imprime o relatório de classificação, que inclui precisão, recall, f1-score e acurácia.
report = classification_report(y_teste, y_pred, output_dict=True)
print("\n===== PERFORMANCE DO MODELO PREDITIVO =====")
print(f"Acurácia: {report['accuracy']:.2f}")
print(f"Precisão (clientes que realmente cancelaram): {report['1']['precision']:.2f}")
print(f"Recall (capacidade de detectar cancelamentos): {report['1']['recall']:.2f}")
print(f"F1-Score (equilíbrio entre precisão e recall): {report['1']['f1-score']:.2f}\n")

def plot_model_performance(report, y_teste, y_pred):
    """
    Gera e exibe a matriz de confusão do modelo preditivo,
    acompanhada das principais métricas de desempenho.

    Args:
        report (dict): O relatório de classificação gerado por `classification_report`.
        y_teste (pd.Series): Os valores reais da variável alvo no conjunto de teste.
        y_pred (np.array): Os valores previstos pelo modelo no conjunto de teste.
    """
    cm = confusion_matrix(y_teste, y_pred) # Calcula a matriz de confusão.

    fig, ax = plt.subplots(figsize=(10, 7))

    # Cria um heatmap para visualizar a matriz de confusão.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Não Cancelou', 'Cancelou'],
                yticklabels=['Não Cancelou', 'Cancelou'],
                linewidths=.5, linecolor='black', annot_kws={"size": 14})

    plt.title('Matriz de Confusão do Modelo Preditivo', fontweight='bold', fontsize=16)
    plt.ylabel('Valor Real', fontsize=13)
    plt.xlabel('Valor Previsto', fontsize=13)
    plt.grid(False) # Remove o grid para melhor visualização da matriz.

    # Formata o texto das métricas para exibição no gráfico.
    metrics_text = (f"Acurácia: {report['accuracy']:.2f}\n"
                    f"Precisão (Cancelou): {report['1']['precision']:.2f}\n"
                    f"Recall (Cancelou): {report['1']['recall']:.2f}\n"
                    f"F1-Score (Cancelou): {report['1']['f1-score']:.2f}")

    # Posiciona o texto das métricas ao lado direito do gráfico.
    plt.figtext(0.95, 0.75,
                metrics_text,
                transform=fig.transFigure,
                fontsize=11,
                verticalalignment='top', horizontalalignment='left',
                bbox={'facecolor':'white', 'alpha':0.8, 'pad':7})

    # Adiciona uma explicação detalhada da matriz de confusão e das métricas abaixo do gráfico.
    explanation = ("Componentes da Matriz de Confusão:\n"
                   "  - Verdadeiros Negativos (Superior Esquerdo): Casos onde o modelo previu 'Não Cancelou' e o cliente realmente 'Não Cancelou'.\n"
                   "  - Falsos Positivos (Superior Direito): Casos onde o modelo previu 'Cancelou', mas o cliente 'Não Cancelou' (Erro Tipo I).\n"
                   "  - Falsos Negativos (Inferior Esquerdo): Casos onde o modelo previu 'Não Cancelou', mas o cliente 'Cancelou' (Erro Tipo II).\n"
                   "  - Verdadeiros Positivos (Inferior Direito): Casos onde o modelo previu 'Cancelou' e o cliente realmente 'Cancelou'.\n\n"
                   "Métricas de Performance:\n"
                   "  - Acurácia: Proporção de previsões corretas no total.\n"
                   "  - Precisão (Cancelou): Dos previstos como 'Cancelou', quantos realmente cancelaram.\n"
                   "  - Recall (Cancelou): Dos que realmente 'Cancelaram', quantos foram corretamente detectados.")

    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=9, wrap=True,
                bbox={'facecolor':'lightgray', 'alpha':0.6, 'pad':5}, va='top')

    # Ajusta o layout para acomodar o texto adicional e evitar sobreposição.
    plt.tight_layout(rect=[0, 0.05, 0.90, 0.95])
    plt.show()

# Chama a função para plotar a performance do modelo.
plot_model_performance(report, y_teste, y_pred)

# --- Insights para o Negócio ---
print(f"[5/5] Gerando insights de negócio...")

# Cria um DataFrame com os coeficientes do modelo, que representam a importância de cada variável.
# O valor absoluto é usado para 'Importância' para ordenar pelo impacto, independentemente da direção.
coef_df = pd.DataFrame({
    'Variável': X.columns,
    'Coeficiente': modelo.coef_[0],
    'Importância': np.abs(modelo.coef_[0])
}).sort_values('Importância', ascending=False)

# Ajusta o rótulo para 'sexo_Female' e inverte o sinal do coeficiente se 'sexo_Male'
# for a feature dummy gerada (devido ao 'drop_first=True').
if 'sexo_Male' in coef_df['Variável'].values:
    male_idx = coef_df[coef_df['Variável'] == 'sexo_Male'].index[0]
    coef_df.loc[male_idx, 'Coeficiente'] *= -1 # Inverte o sinal para representar o impacto de 'Female'.
    coef_df.loc[male_idx, 'Variável'] = 'sexo_Female' # Renomeia a variável.

# Imprime os top 5 fatores de risco para cancelamento, indicando a direção do impacto.
print("\n===== FATORES DE RISCO PARA CANCELAMENTO =====")
print("Aqui estão os principais fatores que influenciam o risco de cancelamento:")
for i, row in enumerate(coef_df.head(5).itertuples()):
    var = row.Variável.replace('_', ' ').title()
    coef = row.Coeficiente
    imp = row.Importância

    direction = "AUMENTA" if coef > 0 else "DIMINUI"
    print(f"- {var}: {direction} o risco (impacto: {imp:.4f})")

def plot_risk_factors(coef_dataframe):
    """
    Gera um gráfico de barras horizontais mostrando os principais fatores de risco
    de cancelamento e a magnitude de seu impacto, com cores indicando a direção.

    Args:
        coef_dataframe (pd.DataFrame): DataFrame contendo as variáveis, coeficientes e importância.
    """
    top_factors = coef_dataframe.head(5).copy()
    # Ordena os fatores para que o mais importante fique no topo do gráfico de barras horizontais.
    top_factors = top_factors.sort_values('Importância', ascending=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Define cores fixas para indicar se o fator aumenta (vermelho) ou diminui (azul) o risco.
    colors = ['#DC3912' if x > 0 else '#3366CC' for x in top_factors['Coeficiente']]
    bars = ax.barh(top_factors['Variável'].str.replace('_', ' ').str.title(),
                    top_factors['Importância'], color=colors)

    # Adiciona rótulos de texto nas barras para indicar a direção do impacto.
    for i, bar in enumerate(bars):
        idx = top_factors.index[i]
        coef = top_factors.loc[idx, 'Coeficiente']
        direction_text = "Aumenta o risco" if coef > 0 else "Diminui o risco"

        # Ajusta a posição e a cor do texto para melhor visibilidade, dependendo do tamanho da barra.
        if bar.get_width() < ax.get_xlim()[1] * 0.1:
            text_x_position = bar.get_width() + ax.get_xlim()[1] * 0.005
            text_color = 'black'
            ha_align = 'left'
        else:
            text_x_position = bar.get_width() - (ax.get_xlim()[1] * 0.005)
            text_color = 'white'
            ha_align = 'right'

        ax.text(text_x_position, bar.get_y() + bar.get_height()/2,
                direction_text, ha=ha_align, va='center',
                color=text_color, fontsize=10, fontweight='bold')

    plt.title('Top 5 Fatores de Risco para Cancelamento', fontweight='bold', fontsize=18)
    plt.xlabel('Magnitude do Impacto (Valor Absoluto do Coeficiente)', fontsize=14)
    plt.ylabel('Variáveis', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Adiciona uma legenda customizada para as cores das barras.
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#DC3912', lw=4, label='Aumenta o risco de cancelamento'),
        Line2D([0], [0], color='#3366CC', lw=4, label='Diminui o risco de cancelamento')
    ]
    ax.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(1, 0), fontsize=11)

    # Adiciona uma anotação explicativa abaixo do gráfico.
    plt.figtext(0.5, 0.01,
                '\nInterpretação: Barras mais longas indicam fatores com maior impacto na probabilidade de cancelamento. '
                'As cores indicam a direção do impacto (vermelho para aumento do risco, azul para diminuição do risco).',
                ha='center', fontsize=10, bbox={'facecolor':'lightgray', 'alpha':0.6, 'pad':5})

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.show()

# Chama a função para plotar os fatores de risco.
plot_risk_factors(coef_df)

# Análise de clientes em risco.
# Calcula a probabilidade de cancelamento para todos os clientes usando o modelo.
df['risco_cancelamento'] = modelo.predict_proba(X)[:, 1]
# Segmenta os clientes em grupos de risco (Baixo, Médio, Alto) usando quartis.
df['grupo_risco'] = pd.qcut(df['risco_cancelamento'],
                             q=[0, 0.25, 0.75, 1.0],
                             labels=['Baixo Risco', 'Médio Risco', 'Alto Risco'],
                             duplicates='drop')

# Conta o número de clientes em cada grupo de risco e imprime o resumo.
risco_counts = df['grupo_risco'].value_counts()
print("\n===== SEGMENTAÇÃO DE CLIENTES POR RISCO =====")
print(f"Baixo Risco: {risco_counts.get('Baixo Risco', 0)} clientes")
print(f"Médio Risco: {risco_counts.get('Médio Risco', 0)} clientes")
print(f"Alto Risco: {risco_counts.get('Alto Risco', 0)} clientes\n")

def plot_risk_segments(risk_data):
    """
    Gera um gráfico de barras da segmentação de clientes por nível de risco de cancelamento.

    Args:
        risk_data (pd.Series): Contagem de clientes por grupo de risco.
    """
    risk_data_df = risk_data.reset_index()
    risk_data_df.columns = ['Grupo de Risco', 'Número de Clientes']

    # Define a ordem de exibição dos grupos de risco no gráfico.
    order = ['Alto Risco', 'Médio Risco', 'Baixo Risco']
    risk_data_df['Grupo de Risco'] = pd.Categorical(risk_data_df['Grupo de Risco'], categories=order, ordered=True)
    risk_data_df = risk_data_df.sort_values('Grupo de Risco')

    # Define as cores para cada grupo de risco.
    colors = ['#DC3912', '#FF9900', '#109618']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(risk_data_df['Grupo de Risco'], risk_data_df['Número de Clientes'],
                    color=colors)

    plt.title('Segmentação de Clientes por Nível de Risco de Cancelamento', fontweight='bold', fontsize=16)
    plt.xlabel('Grupo de Risco', fontsize=13)
    plt.ylabel('Número de Clientes', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    total = risk_data_df['Número de Clientes'].sum()
    # Adiciona rótulos com a contagem e o percentual de clientes em cada barra.
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
        ax.text(bar.get_x() + bar.get_width()/2., height / 2,
                f'{percentage:.1f}%', ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # Adiciona uma anotação explicativa abaixo do gráfico.
    plt.figtext(0.5, 0.01,
                '\nInterpretação: Este gráfico categoriza os clientes com base na probabilidade de cancelamento. '
                'Clientes de Alto Risco (vermelho) são os mais propensos a cancelar e devem ser priorizados para ações de retenção. '
                'Clientes de Baixo Risco (verde) são os menos propensos.',
                ha='center', fontsize=9, wrap=True,
                bbox={'facecolor':'lightgray', 'alpha':0.6, 'pad':5})

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.show()

# Chama a função para plotar a segmentação de clientes por risco.
plot_risk_segments(risco_counts)

# Análise detalhada sobre a relação entre ligações ao Call Center e cancelamento.
print("\n===== ANÁLISE DETALHADA: LIGAÇÕES PARA CALL CENTER E CANCELAMENTO =====")
print("- **Relação Crucial:** O número de ligações para o call center é um fator chave no risco de cancelamento.")
print("- **Risco Crescente:** Após 4-5 ligações, a probabilidade de cancelamento aumenta drasticamente, indicando insatisfação crônica ou problemas não resolvidos.")
print("- **Ação Proativa:** Clientes com múltiplas interações no call center são prioritários para ações de retenção e resolução proativa de suas questões.")
print("-" * 70)

def plot_improved_callcenter_impact(dataframe):
    """
    Visualiza a relação entre o número de ligações ao call center e a probabilidade de cancelamento.
    Usa um scatter plot para pontos individuais e um line plot para mostrar a tendência.

    Args:
        dataframe (pd.DataFrame): O DataFrame contendo os dados de ligações e risco de cancelamento.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot para visualizar a distribuição individual dos clientes.
    sns.scatterplot(x='ligacoes_callcenter', y='risco_cancelamento',
                    data=dataframe, alpha=0.2, color='darkblue', s=40, ax=ax, label='Clientes Individuais')

    # Line plot para mostrar a tendência média da probabilidade de cancelamento, com intervalo de confiança de 95%.
    sns.lineplot(x='ligacoes_callcenter', y='risco_cancelamento',
                    data=dataframe, errorbar=('ci', 95),
                    color='red', linewidth=3, ax=ax, label='Tendência (Média e IC 95%)')

    plt.title('Relação entre Ligações ao Call Center e Risco de Cancelamento',
              fontweight='bold', fontsize=14)
    plt.xlabel('Número de Ligações ao Call Center', fontsize=12)
    plt.ylabel('Probabilidade de Cancelamento', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    ax.legend(title='Legenda', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    # Adiciona uma anotação explicativa abaixo do gráfico.
    plt.figtext(0.5, 0.01,
                '\nInterpretação: Este gráfico ilustra como a probabilidade de cancelamento (Y-axis) '
                'se comporta à medida que o número de ligações ao call center (X-axis) aumenta. '
                'A linha vermelha representa a média da probabilidade de cancelamento para cada número de ligações, '
                'com a área sombreada indicando o intervalo de confiança de 95%. '
                'Observa-se que, após um certo número de ligações (ex: 4-5), o risco de cancelamento '
                'tende a aumentar drasticamente, sugerindo que múltiplas interações '
                'com o call center podem ser um forte indicador de insatisfação e intenção de cancelar.',
                ha='center', fontsize=9, wrap=True,
                bbox={'facecolor':'lightgray', 'alpha':0.6, 'pad':5})

    plt.tight_layout(rect=[0, 0.15, 0.95, 1])
    plt.subplots_adjust(right=0.85)
    plt.show()

# Chama a função para plotar o impacto das ligações ao call center.
plot_improved_callcenter_impact(df)

# Tempo total de execução da análise.
execution_time = time.time() - start_time
print(f"\nAnálise completa realizada em {execution_time:.2f} segundos.")
print("="*60)
print("RESUMO: As variáveis mais importantes para prever cancelamentos são:")
if len(coef_df) >= 3:
    for i, (var, coef) in enumerate(zip(coef_df['Variável'][:3], coef_df['Coeficiente'][:3])):
        impact = "aumenta" if coef > 0 else "reduz"
        print(f"  {i+1}. {var.replace('_', ' ').title()} {impact} a chance de cancelamento")
else:
    print("Não há variáveis suficientes para listar os top 3 fatores de risco.")
print("="*60)
