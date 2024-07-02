from classes.Dataset_class import Dataset
import json

with open("../config/config.json", "r") as config_file:
    config = json.load(config_file)

dataset_name = config["dataset_name"]
target_feature_name = config["target_feature"]

Dataset = Dataset(dataset_name, target_feature_name);