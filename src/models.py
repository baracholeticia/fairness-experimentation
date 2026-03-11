from sklearn.tree import DecisionTreeClassifier

def get_model():
    #modelo de árvore de decisão
    return DecisionTreeClassifier(random_state=42)