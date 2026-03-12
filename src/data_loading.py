import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import numpy as np

# função que realiza pre-processamento e retorna a dataframe
def preprocess_dataset(df):
    # Primeiro são removidas as colunas que não ajudam na predição
    cols_to_drop = [
        'NOME_BANCO',  # apresentou muitos nulos e irrelevante
        'NOME_TITULAR',  # info pessoal
        'NOME_PESSOA_OD',  # info pessoal
        'CPF_CNPJ_TITULAR',  # info pessoal
        'CPF_CNPJ_OD',  # info pessoal
        'NUMERO_CONTA_OD',  # identificação
        'NUMERO_CONTA'  # identificação
    ]
    df = df.drop(columns=cols_to_drop)

    # tratando valores nulos

    #agora, o objetivo é padronizar os valores das colunas numericas extremas entre si para não comprometer o treinamento
    num_cols = ['VALOR_TRANSACAO', 'VALOR_SALDO']
    scaler = StandardScaler()  #usa-se standardscaler para não distorcer a escala das features
    df[num_cols] = scaler.fit_transform(df[num_cols])

    #tratando variaveis categoricas
    # CNAB: foram observadas 4 possibilidades, então será usado one-hot encoding
    # NATUREZA_LANCAMENTO: D ou C, poderia ser label mas será OHE por ter mesmo efeito na prática nesse caso
    cat_cols = ['CNAB', 'NATUREZA_LANCAMENTO']
    ohe = OneHotEncoder(sparse_output=False, drop='first')  #evita colinearidade no OHE
    cat_encoded = ohe.fit_transform(df[cat_cols])
    cat_encoded_df = pd.DataFrame(cat_encoded, columns=ohe.get_feature_names_out(cat_cols))

    #removendo col originais categóricas e adicionando codificadas
    df = df.drop(columns=cat_cols)
    df = pd.concat([df.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)

    #RAMO_ATIVIDADE_1 não será usado como input como denotado no comando do projeto,sendo mantido para métricas de fairness
    protected_col = df['RAMO_ATIVIDADE_1']

    # Retornar dataset pré-processado e atributo protegido separado
    return df, protected_col # por ora não sera usado esse segundo retorno


path = "../data/raw"

datasets = []

for file in os.listdir(path):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(path, file), low_memory=False)
        df_clean, protected = preprocess_dataset(df)
        df_clean.to_csv(f"../data/processed/{file.replace(".csv", "")}_processed.csv", index=False)
        print(df_clean.head())
        print(protected.value_counts())  # distribuição do grupo privilegiado vs não privilegiado
                                         # CASO EU VÁ REPONDERAR
        datasets.append(df_clean)


