from aif360.metrics import ClassificationMetric

#grupo privilegiado : 1, grupo desprivilegiado: 4
privileged_groups = [{'RAMO_ATIVIDADE': 1}]
unprivileged_groups = [{'RAMO_ATIVIDADE': 4}]

def evaluate_fairness(dataset_test, predictions):
    #dataset com previões para comparar o real com o predito
    dataset_predicted = dataset_test.copy()
    dataset_predicted.labels = predictions.reshape(-1, 1)
    
    #inicializa a classe de métricas do aif360
    metrics = ClassificationMetric(
        dataset_test, 
        dataset_predicted,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    #extrai as 4 métricas que foram utilizadas no artigo 
    stat_parity = metrics.statistical_parity_difference()
    avg_odds = metrics.average_abs_odds_difference()
    equal_opp = metrics.equal_opportunity_difference()
    disp_impact = metrics.disparate_impact()
    
    #transforma em valores absolutos
    return {
        'Statistical Parity (Abs)': abs(stat_parity),
        'Average Abs Odds': abs(avg_odds),
        'Equal Opportunity (Abs)': abs(equal_opp),
        'Disparate Impact': min(abs(1 - disp_impact), 1.0) #normaliza a distancia para 1
    }