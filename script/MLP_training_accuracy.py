import json
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from initialization.initialization_Dataset import Dataset


simplefilter("ignore", category=ConvergenceWarning)

# Crea un classificatore MLP con i parametri predefiniti
mlp = MLPClassifier()

# Definisci i parametri da testare durante la ricerca della griglia
parameters = {
    'max_iter': (100, 200, 300),
    'hidden_layer_sizes': [(10,30,10),(20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

# Applica la ricerca della griglia utilizzando GridSearchCV
gs = GridSearchCV(mlp, parameters)

# Addestra il modello utilizzando i dati di addestramento e le etichette di addestramento
gs.fit(Dataset.x_train, Dataset.y_train)

# Salva il miglior modello ottenuto dalla ricerca della griglia in una variabile chiamata final_model
final_model = gs.best_estimator_

# Predirre le classi di x_test
labels = final_model.predict(Dataset.x_test)

# Calcolare l'accuratezza delle previsioni
accuracy = accuracy_score(Dataset.y_test, labels)
print("Accuracy:", accuracy)

# Salvo i parametri del modello
config_path = 'config/modelConfiguration.json'
with open(config_path, 'w') as f:
    json.dump(gs.best_params_, f)

