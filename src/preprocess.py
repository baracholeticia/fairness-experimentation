import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from feature_engineering import run_feature_engineering

#TODO: Leitura com o carregamento do atributo protegido e das saidas @Leticia_Baracho
# X    = pd.read_parquet('data/processed/pd_v1_X.parquet')
# y    = pd.read_parquet('data/processed/pd_v1_y.parquet').squeeze()
# prot = pd.read_parquet('data/processed/pd_v1_prot.parquet').squeeze()

path = "../data/raw"
target_path = "../data/processed"

# realiza feature engineering e pré-processamento
def preprocess_dataset(
        df: pd.DataFrame):  # talvez melhor fazer tudo em memoria sem salvar o processamento msm
    # agregando antes de descartar identificadores
    df = run_feature_engineering(df)

   #devido aos problemas de hardware
    drop_cols = ['I-e', 'CNAB', 'NATUREZA_LANCAMENTO']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # usando standardscaler para não distorcer a escala relativa
    num_cols = [c for c in [
        'VALOR_TRANSACAO', 'VALOR_SALDO',
        'valor_total_dia', 'std_valor_dia', 'ticket_medio_dia',
        'valor_total_mes', 'velocidade_gasto_mes',
        'ratio_transacao_saldo', 'concentracao_cnpj_mes',
        'desvio_valor_vs_media_dia',
        'grau_saida_global',
    ] if c in df.columns]



    if num_cols:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

    # separando atributo protegido
    protected_col = df.pop('RAMO_ATIVIDADE_1') if 'RAMO_ATIVIDADE_1' in df.columns else None

    # separando POSSIVEL target TODO: Verificar se é mesmo
    y = df.pop('I-d')

    return df, y, protected_col

#função que foi chamada para salvar os dados processados
def process_and_save():
    os.makedirs(target_path, exist_ok=True)

    for file in sorted(os.listdir(path)):
        if not file.endswith(".csv"):
            continue

        raw_path = os.path.join(path, file)
        stem = os.path.splitext(file)[0]

        print(f"Processando {file} ...", end=" ", flush=True)
        raw = pd.read_csv(raw_path, low_memory=False)

        x, y, protected = preprocess_dataset(raw)

        x.to_parquet(os.path.join(target_path, f"{stem}_X.parquet"), index=False)
        y.to_frame().to_parquet(os.path.join(target_path, f"{stem}_y.parquet"), index=False)
        protected.to_frame().to_parquet(
            os.path.join(target_path, f"{stem}_prot.parquet"), index=False)

        print(f"shape={x.shape}  ->  salvo em {target_path}/{stem}_*.parquet")

