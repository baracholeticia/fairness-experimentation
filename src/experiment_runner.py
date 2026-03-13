import pandas as pd
import numpy as np
import os
import gc
import warnings
from aif360.datasets import BinaryLabelDataset

from models import get_model
from fairness_metrics import evaluate_fairness

warnings.filterwarnings("ignore")

TARGET_COL = 'I-d'
PROTECTED_COL = 'RAMO_ATIVIDADE' 

def prepare_aif360_dataset(df, features_to_keep):

    # isola o grupo privilegiado e o desprivilegiado
    df_filtered = df[df[PROTECTED_COL].isin([1, 4])].copy()
    
    df_filtered[PROTECTED_COL] = df_filtered[PROTECTED_COL].map({1: 1.0, 4: 0.0})
    
    # deixa apenas as features da iteração atual
    cols_to_keep = list(set(features_to_keep + [PROTECTED_COL, TARGET_COL]))
    df_subset = df_filtered[cols_to_keep].copy()
    
    # garante que alvo e protegido sao numericos
    df_subset[TARGET_COL] = pd.to_numeric(df_subset[TARGET_COL], errors='coerce').fillna(0.0)
    df_subset[PROTECTED_COL] = pd.to_numeric(df_subset[PROTECTED_COL], errors='coerce').fillna(0.0)
    
    # criação do objeto nativo do aif360
    return BinaryLabelDataset(
        favorable_label=0.0,
        unfavorable_label=1.0,
        df=df_subset,
        label_names=[TARGET_COL],
        protected_attribute_names=[PROTECTED_COL],
        privileged_protected_attributes=[[1.0]],
        unprivileged_protected_attributes=[[0.0]]
    )

def run_experiment():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    processed_path = os.path.join(dir_path, "..", "data", "processed")
    datasets = []
    nomes_bases = [] 

    arquivos_x = sorted([f for f in os.listdir(processed_path) if f.endswith('_X.parquet')])
    prefixos = [f.replace('_X.parquet', '') for f in arquivos_x]

    for pref in prefixos:
        X = pd.read_parquet(os.path.join(processed_path, f"{pref}_X.parquet"))
        y = pd.read_parquet(os.path.join(processed_path, f"{pref}_y.parquet"))
        prot = pd.read_parquet(os.path.join(processed_path, f"{pref}_prot.parquet"))
        
        df = X.copy()
        df[TARGET_COL] = y.values
        df[PROTECTED_COL] = prot.values

        idx = (
            df.groupby(PROTECTED_COL, group_keys=False)
            .apply(lambda x: x.sample(frac=0.3, random_state=42))
            .index
        )
        df = df.loc[idx].reset_index(drop=True)
        
        datasets.append(df)
        nomes_bases.append(pref)

    todas_colunas = datasets[0].select_dtypes(include=['number']).columns.tolist()
    
    for col in [TARGET_COL, PROTECTED_COL, 'RAMO_ATIVIDADE_1', 'I-e']:
        if col in todas_colunas:
            todas_colunas.remove(col)
    
    base_features = [todas_colunas.pop(0), todas_colunas.pop(0)] 
    colunas_para_testar = todas_colunas 

    resultados_detalhados = [] 

    for i in range(len(colunas_para_testar) + 1):
        features_atuais = base_features + colunas_para_testar[:i]
        tamanho_conjunto = len(features_atuais) + 1 

        print(f"teste com {tamanho_conjunto} features")
        
        ciclos_resultados_temp = [] 
        
        for c in range(len(datasets)):
            df_train_raw = datasets[c]
            nome_base_treino = nomes_bases[c]
            df_test_raw = pd.concat([datasets[j] for j in range(len(datasets)) if j != c], ignore_index=True)
            
            # converao pra aif360
            dataset_train_aif360 = prepare_aif360_dataset(df_train_raw, features_atuais)
            dataset_test_aif360 = prepare_aif360_dataset(df_test_raw, features_atuais)
            
            # etração do dado e treinamento do modelo
            X_train = dataset_train_aif360.features
            y_train = dataset_train_aif360.labels.ravel()
            X_test = dataset_test_aif360.features
            
            model = get_model()
            model.fit(X_train, y_train)
            
            # previsão
            predictions = model.predict(X_test)
            fairness_results = evaluate_fairness(dataset_test_aif360, predictions)
            
            registro_ciclo = {
                'Num_Features': tamanho_conjunto,
                'Ciclo': c + 1,
                'Base_Treino': nome_base_treino, 
                'Accuracy': fairness_results['Accuracy'],
                'Statistical Parity (Abs)': fairness_results['Statistical Parity (Abs)'],
                'Average Abs Odds': fairness_results['Average Abs Odds'],
                'Equal Opportunity (Abs)': fairness_results['Equal Opportunity (Abs)'],
                'Disparate Impact': fairness_results['Disparate Impact']
            }
            
            resultados_detalhados.append(registro_ciclo)
            ciclos_resultados_temp.append(registro_ciclo)

            del df_train_raw, df_test_raw, dataset_train_aif360, dataset_test_aif360, X_train, y_train, X_test
            gc.collect()

        media_acc = np.mean([res['Accuracy'] for res in ciclos_resultados_temp])
        media_sp = np.mean([res['Statistical Parity (Abs)'] for res in ciclos_resultados_temp])
        media_ao = np.mean([res['Average Abs Odds'] for res in ciclos_resultados_temp])
        media_eo = np.mean([res['Equal Opportunity (Abs)'] for res in ciclos_resultados_temp])
        media_di = np.mean([res['Disparate Impact'] for res in ciclos_resultados_temp])
        
        print(f"MÉDIA DOS 10 CICLOS: Acurácia: {media_acc:.3f} | Stat Parity: {media_sp:.3f} | Avg Odds: {media_ao:.3f} | Equal Opp: {media_eo:.3f} | Disp Impact: {media_di:.3f}")

    df_final = pd.DataFrame(resultados_detalhados)
    cols = ['Num_Features', 'Ciclo', 'Base_Treino', 'Accuracy', 'Statistical Parity (Abs)', 'Average Abs Odds', 'Equal Opportunity (Abs)', 'Disparate Impact']
    df_final = df_final[cols]

    caminho_saida = os.path.join(dir_path, "..", "resultados.csv")
    df_final.to_csv(caminho_saida, index=False)
if __name__ == "__main__":
    run_experiment()