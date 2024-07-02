from initialization.initialization_CustomIsolationForest import CIF_model
from initialization.initialization_Dataset import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print(CIF_model)

print(CIF_model.predict_proba(Dataset.x_test))

print(CIF_model.predict(Dataset.x_test))
print(Dataset.y_test)
print(CIF_model.score(Dataset.x_test, Dataset.y_test))

y_pred = CIF_model.predict(Dataset.x_test)
print(y_pred)

