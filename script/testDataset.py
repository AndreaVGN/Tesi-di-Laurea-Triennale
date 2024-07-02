from initialization.initialization_CustomIsolationForest import CIF_model
from initialization.initialization_Dataset import Dataset
from script.MLPClassifier import clf

Dataset.get_shapley_values(clf)
Dataset.get_shapley_values(CIF_model)
Dataset.get_greedy_chart(clf, 0)
Dataset.get_greedy_chart(CIF_model, 0)
Dataset.get_counterplots(clf, 0)
Dataset.get_counterplots(CIF_model, 0)

CIF_model.test_spearman(Dataset);