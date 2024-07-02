from cfnow import find_tabular
from sklearn.model_selection import train_test_split
import pandas as pd
import shap
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but MLPClassifier was fitted with feature names")
warnings.filterwarnings("ignore", message="X does not have valid feature names, but CustomIsolationForest was fitted with feature names")

class Dataset:
    def __init__(self, dataset_name, target_feature_name):
        self.dataset = pd.read_csv("../resource/" + dataset_name)
        self.target = self.dataset[target_feature_name]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset,
                                                                                self.target,
                                                                                test_size=0.4,
                                                                                random_state=42)
        self.x_train = self.x_train.drop(target_feature_name, axis=1)
        self.x_test = self.x_test.drop(target_feature_name, axis=1)

    def get_dataset_train(self):
        return pd.concat([self.x_train, self.y_train], axis=1)

    def get_dataset_test(self):
        return pd.concat([self.x_test, self.y_test], axis=1)

    def get_X(self):
        return self.dataset.drop('ANOMALOUS', axis=1)

    def get_shapley_values(self, model):
        # Creazione di un oggetto explainer di SHAP utilizzando il modello e il metodo KernelExplainer
        explainer = shap.KernelExplainer(model.predict_proba, self.x_train)
        # Calcolo degli Shapley Values per un set di dati di test
        shap_values = explainer.shap_values(self.x_test)
        # Visualizzazione degli Shapley Values
        shap.summary_plot(shap_values, self.x_test)

    def get_greedy_chart(self, model, row):
        cols_numericals = [column for column in self.dataset.select_dtypes(include=['int64']).columns]

        cf_res = find_tabular(
            factual=self.x_test.iloc[row],
            feat_types={c: 'num' if c in cols_numericals else 'cat' for c in self.get_X().columns},
            has_ohe=True,
            model_predict_proba=model.predict_proba,
            limit_seconds=1)  # 60

        cf_res.generate_counterplots(0).greedy('../output/greedy.png')

    def get_counterplots(self, model, row):
        cols_numericals = [column for column in self.dataset.select_dtypes(include=['int64']).columns]

        cf_res = find_tabular(
            factual=self.x_test.iloc[row],
            feat_types={c: 'num' if c in cols_numericals else 'cat' for c in self.get_X().columns},
            has_ohe=True,
            model_predict_proba=model.predict_proba,
            limit_seconds=1)  # 60

        cf_res.generate_counterplots(0).countershapley('../output/countershapley.png')

