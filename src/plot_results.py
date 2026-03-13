import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_experiment_results():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(dir_path, "..", "resultados.csv")
    
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")

    # 1- média geral da acurácia e métricas de justiça
    print("gerando gráfico 1")
    plt.figure(figsize=(12, 7))
    
    # agrupa por número de features e tira a média dos ciclos
    df_mean = df.groupby('Num_Features').mean(numeric_only=True).reset_index()

    plt.plot(df_mean['Num_Features'], df_mean['Accuracy'], marker='o', linewidth=2, label='Acurácia')
    plt.plot(df_mean['Num_Features'], df_mean['Disparate Impact'], marker='s', linewidth=2, label='Disparate Impact (Abs)')
    plt.plot(df_mean['Num_Features'], df_mean['Average Abs Odds'], marker='^', linewidth=2, label='Average Abs Odds')
    plt.plot(df_mean['Num_Features'], df_mean['Statistical Parity (Abs)'], marker='d', linewidth=2, label='Statistical Parity')

    plt.title('Evolução Média: Acurácia vs. Métricas de Injustiça pelo Tamanho do Conjunto de Features', fontsize=14)
    plt.xlabel('Número de Features', fontsize=12)
    plt.ylabel('Valor da Métrica (0.0 a 1.0)', fontsize=12)
    plt.xticks(range(int(df['Num_Features'].min()), int(df['Num_Features'].max()) + 1))
    plt.legend()
    plt.tight_layout()
    
    out_path_1 = os.path.join(dir_path, "..", 'grafico_1_evolucao_media.png')
    plt.savefig(out_path_1, dpi=300)
    plt.close()

    # 2- boxplot do disparate impact
    print("gerando gráfico 2")
    plt.figure(figsize=(12, 7))
    
    sns.boxplot(data=df, x='Num_Features', y='Disparate Impact', palette="Blues")
    sns.stripplot(data=df, x='Num_Features', y='Disparate Impact', color='red', alpha=0.6, jitter=True)

    plt.title('Estabilidade das Previsões: Variação do Disparate Impact nos 10 Ciclos', fontsize=14)
    plt.xlabel('Número de Features', fontsize=12)
    plt.ylabel('Disparate Impact', fontsize=12)
    plt.tight_layout()
    
    out_path_2 = os.path.join(dir_path, "..", 'grafico_2_boxplot_estabilidade.png')
    plt.savefig(out_path_2, dpi=300)
    plt.close()

    # 3-impacto do nivel de viés da base
    print("gerando gráfico 3")
    plt.figure(figsize=(14, 8))
    
    # traça uma linha para cada uma das 10 bases de treino
    sns.lineplot(data=df, x='Num_Features', y='Disparate Impact', hue='Base_Treino', marker='o', palette="tab10")

    plt.title('Impacto do Nível de Viés: Disparate Impact Evoluindo por Base Específica', fontsize=14)
    plt.xlabel('Número de Features', fontsize=12)
    plt.ylabel('Disparate Impact', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Base de Treino")
    plt.tight_layout()
    
    out_path_3 = os.path.join(dir_path, "..", 'grafico_3_evolucao_por_base.png')
    plt.savefig(out_path_3, dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_experiment_results()