from classes.CustomIsolationForest_class import CustomIsolationForest
from initialization.initialization_Dataset import Dataset

CIF_model = CustomIsolationForest()

CIF_model = CIF_model.fit(Dataset.x_train, Dataset.y_train)

CIF_model.init_predict_proba(Dataset.x_train)

















