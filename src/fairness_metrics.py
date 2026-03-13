from aif360.metrics import ClassificationMetric

# grupo privilegiado =1, desprivilegiado =4
privileged_groups = [{'RAMO_ATIVIDADE': 1.0}]
unprivileged_groups = [{'RAMO_ATIVIDADE': 0.0}]

def evaluate_fairness(dataset_test, predictions):
    # cria uma cópia para receber as previsões
    dataset_predicted = dataset_test.copy(deepcopy=True)
    dataset_predicted.labels = predictions.reshape(-1, 1)
    
    metrics = ClassificationMetric(
        dataset_test, 
        dataset_predicted,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    # extrai as métricas
    return {
        'Accuracy': metrics.accuracy(),
        'Statistical Parity (Abs)': abs(metrics.statistical_parity_difference()),
        'Average Abs Odds': abs(metrics.average_abs_odds_difference()),
        'Equal Opportunity (Abs)': abs(metrics.equal_opportunity_difference()),
        'Disparate Impact': min(abs(1 - metrics.disparate_impact()), 1.0)
    }