import pandas as pd


# para extração de features temporais
def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai features temporais a partir de DATA_LANCAMENTO.
    Requer que a coluna já esteja em formato datetime.
    """
    df = df.copy()
    df['DATA_LANCAMENTO'] = pd.to_datetime(df['DATA_LANCAMENTO'], errors='coerce')

    df['dia_semana'] = df['DATA_LANCAMENTO'].dt.dayofweek
    df['mes'] = df['DATA_LANCAMENTO'].dt.month
    df['fim_de_semana'] = (df['DATA_LANCAMENTO'].dt.dayofweek >= 5).astype(int)

    # chave auxiliar para agrupamentos
    df['_data_dia'] = df['DATA_LANCAMENTO'].dt.date
    df['_ano_mes'] = df['DATA_LANCAMENTO'].dt.tz_localize(None).dt.to_period('M').astype(str)
    return df


# cria features de agregação diária e mensal por conta
def build_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # diária
    agg_diaria = (
        df.groupby(['NUMERO_CONTA', '_data_dia'])
        .agg(
            n_transacoes_dia=('VALOR_TRANSACAO', 'count'),
            valor_total_dia=('VALOR_TRANSACAO', 'sum'),
            valor_medio_dia=('VALOR_TRANSACAO', 'mean'),
            std_valor_dia=('VALOR_TRANSACAO', 'std'),
            n_cnpj_od_dia=('CPF_CNPJ_OD', 'nunique'),
            n_contas_od_dia=('NUMERO_CONTA_OD', 'nunique'),
        )
        .reset_index()
    )
    agg_diaria['std_valor_dia'] = agg_diaria['std_valor_dia'].fillna(0)

    # agregação mensal
    agg_mensal = (
        df.groupby(['NUMERO_CONTA', '_ano_mes'])
        .agg(
            n_dias_ativo_mes=('_data_dia', 'nunique'),
            n_transacoes_mes=('VALOR_TRANSACAO', 'count'),
            valor_total_mes=('VALOR_TRANSACAO', 'sum'),
            n_cnpj_od_mes=('CPF_CNPJ_OD', 'nunique'),
            n_contas_od_mes=('NUMERO_CONTA_OD', 'nunique'),
        )
        .reset_index()
    )

    # join de volta
    df = df.merge(agg_diaria, on=['NUMERO_CONTA', '_data_dia'], how='left')
    df = df.merge(agg_mensal, on=['NUMERO_CONTA', '_ano_mes'], how='left')

    return df


# para features de comportamento, depende das agregações
def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de comportamento / anomalia derivadas das agregações.
    Depende das colunas geradas em build_aggregation_features.
    """
    df = df.copy()

    # valor da transação vs saldo atual
    df['ratio_transacao_saldo'] = df['VALOR_TRANSACAO'] / (df['VALOR_SALDO'].abs() + 1)

    # Velocidade de gasto mensal (valor total / dias ativos no mês)
    df['velocidade_gasto_mes'] = (
            df['valor_total_mes'] / (df['n_dias_ativo_mes'] + 1)
    )

    # concentração
    df['concentracao_cnpj_mes'] = 1 / (df['n_cnpj_od_mes'] + 1)

    # ticket médio diário
    df['ticket_medio_dia'] = df['valor_total_dia'] / (df['n_transacoes_dia'] + 1)

    # desvio do valor da transação relativo à média diária
    df['desvio_valor_vs_media_dia'] = (
            (df['VALOR_TRANSACAO'] - df['valor_medio_dia']) /
            (df['std_valor_dia'] + 1)
    )

    return df


# features de conexão
def build_network_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # quantas contas destino distintas esta conta já usou
    grau_saida = (
        df.groupby('NUMERO_CONTA')['NUMERO_CONTA_OD']
        .nunique()
        .rename('grau_saida_global')
    )

    # quantas contas distintas enviaram para esta conta
    grau_entrada = (
        df.groupby('NUMERO_CONTA_OD')['NUMERO_CONTA']
        .nunique()
        .rename('grau_entrada_global')
        .reset_index()
        .rename(columns={'NUMERO_CONTA_OD': 'NUMERO_CONTA'})
    )

    df = df.merge(grau_saida.reset_index(), on='NUMERO_CONTA', how='left')
    df = df.merge(grau_entrada, on='NUMERO_CONTA', how='left')
    df['grau_entrada_global'] = df['grau_entrada_global'].fillna(0)

    return df


# droppa colunas auxiliares/redundantes
def drop_helper_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        #auxiliares
        '_data_dia', '_ano_mes', 'DATA_LANCAMENTO',
        #redundantes
        'n_transacoes_dia', 'n_transacoes_mes',
        'valor_medio_dia', 'n_dias_ativo_mes',
        'n_cnpj_od_dia', 'n_contas_od_dia',
        #baixo sinal
        'mes', 'CNAB_117', 'CNAB_123', 'CNAB_220',
        'grau_entrada_global', 'fim_de_semana',
        #identificadores
        'NUMERO_CONTA', 'CPF_CNPJ_OD', 'NUMERO_CONTA_OD',
        'CPF_CNPJ_TITULAR', 'NOME_BANCO', 'NOME_TITULAR', 'NOME_PESSOA_OD'
    ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


# pipeline de FE
def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = build_temporal_features(df)
    df = build_aggregation_features(df)
    df = build_behavioral_features(df)
    df = build_network_features(df)
    df = drop_helper_cols(df)

    return df
