import json
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from initialization.initialization_Dataset import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

with open('../config/modelConfiguration.json', 'r') as json_file:
    model_params = json.load(json_file)

clf = MLPClassifier(hidden_layer_sizes=model_params['hidden_layer_sizes'],
                    activation=model_params['activation'],
                    solver=model_params['solver'],
                    alpha=model_params['alpha'],
                    learning_rate=model_params['learning_rate']
                    )

clf.fit(Dataset.x_train, Dataset.y_train)

# Predirre le classi di x_test
labels = clf.predict(Dataset.x_test)

# Calcolare l'accuratezza delle previsioni
accuracy = accuracy_score(Dataset.y_test, labels)
print(Dataset.y_test, labels)
print("Accuracy:", accuracy)
