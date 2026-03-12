import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

from feature_engineering import run_feature_engineering

#realiza feature engineering e pré-processamento
def preprocess_dataset(df: pd.DataFrame): #talvez melhor fazer tudo em memoria sem salvar o processamento msm
    #agregando antes de descartar identificadores
    df = run_feature_engineering(df)

    #CNAB: múltiplas categorias -> OHE
    #NATUREZA_LANCAMENTO: D/C ->OHE
    cat_cols = [c for c in ['CNAB', 'NATUREZA_LANCAMENTO'] if c in df.columns]

    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')#drop first para evitar
        # colinearidade
        cat_encoded    = ohe.fit_transform(df[cat_cols])
        cat_encoded_df = pd.DataFrame(
            cat_encoded,
            columns=ohe.get_feature_names_out(cat_cols),
            index=df.index
        )
        df = df.drop(columns=cat_cols)
        df = pd.concat([df, cat_encoded_df], axis=1)


    #usando standardscaler para não distorcer a escala relativa
    num_cols = [c for c in [
        'VALOR_TRANSACAO', 'VALOR_SALDO',
        'valor_total_dia', 'valor_medio_dia', 'std_valor_dia', 'ticket_medio_dia',
        'valor_total_mes', 'velocidade_gasto_mes',
        'ratio_transacao_saldo', 'concentracao_cnpj_mes',
        'desvio_valor_vs_media_dia',
        'grau_saida_global', 'grau_entrada_global',
    ] if c in df.columns]

    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    #separando atributo protegido
    protected_col = df.pop('RAMO_ATIVIDADE_1') if 'RAMO_ATIVIDADE_1' in df.columns else None

    #separando POSSIVEL target TODO: Verificar se é mesmo
    y = df.pop('I-d')

    return df, y, protected_col

# #testando
# path = "../data/raw"
# datasets = []
#
# for file in sorted(os.listdir(path)):
#     if not file.endswith(".csv"):
#         continue
#
#     raw = pd.read_csv(os.path.join(path, file), low_memory=False)
#     df_processed, protected = preprocess_dataset(raw)
#
#     datasets.append({
#         "file":      file,
#         "df":        df_processed,
#         "protected": protected,
#     })
#
#     print(f"[{file}]  shape={df_processed.shape}  colunas={list(df_processed.columns)}")

