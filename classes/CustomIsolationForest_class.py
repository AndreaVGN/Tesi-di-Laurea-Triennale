import csv
import dice_ml
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import accuracy_score
from collections import OrderedDict
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

class CustomIsolationForest(IsolationForest):
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto',
                 max_features=1.0, bootstrap=False, n_jobs=None, random_state=None,
                 verbose=0, warm_start=False,
                 distance_min=None, distance_max=None):
        super().__init__(n_estimators=n_estimators, max_samples=max_samples,
                         contamination=contamination, max_features=max_features,
                         bootstrap=bootstrap, n_jobs=n_jobs, random_state=random_state,
                         verbose=verbose, warm_start=warm_start)
        self.distance_min = distance_min
        self.distance_max = distance_max

    def init_predict_proba(self, values):
        anomaly_score = self.decision_function(values)
        self.distance_max = anomaly_score.max()
        self.distance_min = anomaly_score.min()

    def predict_proba(self, value):
        class_0_probability = (self.decision_function(value) - self.distance_min) / (self.distance_max - self.distance_min)
        class_1_probability = 1 - class_0_probability
        prob_classes = np.vstack((class_0_probability, class_1_probability)).T
        return prob_classes

    def score(self, x_test, y_test):
        y_pred = 1 - (self.predict(x_test) + 1) // 2
        return accuracy_score(y_test, y_pred)

    def test_spearman(self, dataset):
        continuous_features = dataset.x_train.columns.tolist()
        continuous_features = list(continuous_features)

        with open('../output/sperman_pvalue.csv', 'a', newline='') as file:  # w per sovrascrivere, a in append
            writer = csv.writer(file)

            writer.writerow(['DiCe', 'PermutationImportance', 'Spearman', 'P-value'])

            for i in range(0, 10):
                data = dice_ml.Data(dataframe=dataset.get_dataset_train(), continuous_features=continuous_features,
                                    outcome_name='ANOMALOUS')
                model = dice_ml.Model(model=self, backend="sklearn")
                exp = dice_ml.Dice(data, model, method="random")
                query_instances = dataset.x_test[0:]
                imp = exp.global_feature_importance(query_instances)
                summary_importance_list = list(imp.summary_importance.values())
                result = permutation_importance(self, dataset.x_test, dataset.y_test, n_repeats=50, random_state=42)
                feature_names = dataset.x_test.columns.tolist()
                importance_dict = dict(zip(feature_names, result.importances_mean))
                cleaned_data = {key: value.item() for key, value in importance_dict.items()}
                permutation_importance1 = cleaned_data
                perm_sorted = OrderedDict(sorted(permutation_importance1.items()))
                permutation_importance_list = list(perm_sorted.values())
                corr, p_value = spearmanr(summary_importance_list, permutation_importance_list)
                writer.writerow([summary_importance_list, permutation_importance_list, corr, p_value])




    

