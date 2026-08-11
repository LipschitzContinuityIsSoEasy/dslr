#!/usr/bin/env python3

import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import csv
import json

from sklearn.feature_selection import f_classif

# lire une fois paras et stocker
def load_model_params(file_model):
    try:
        with open (file_model, "r") as f:
            all_params = json.load(f)
    except FileNotFoundError:       
        print(f"Erreur : Le fichier '{file_model}' not found")
        exit(1)
    except PermissionError:
        print(f"Erreur : Le fichier '{file_model}' permission denied")
        exit(1)
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier JSON : {e}")
        exit(1)
    except IOError as e:
        print(f"Erreur lors de l'écriture du CSV : {e}")
        exit(1)
    return all_params

# juste lire model pramametresjson 1 fois et stocker dans un data
# # etape:
# # 1. lire test.csv et stocker les 10 best en meme temps !
# j'ai juste besoin index et les cours, que des chiffres
def load_test_set(all_params, test_file):
    try:
        all_test_set = pd.read_csv(test_file,
                                   usecols= ['Index'] + all_params["best_features"])
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{test_file}' est introuvable.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Erreur : Le fichier '{test_file}' est vide.")
        exit(1)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV : {e}")
        exit(1)
    # print(all_test_set)
    if all_test_set['Index'].isnull().any():
        print("Attention : Il y a des valeurs manquantes dans la colonne Index !")

    # Z-score normalization
    for col in all_params["best_features"]:
        train_means = all_params["train_means"][col]
        train_stds = all_params["train_stds"][col]
        if train_stds == 0:
            train_stds = 1.0
        
        all_test_set[col] = all_test_set[col].fillna(train_means)

        all_test_set[col] = (all_test_set[col] - train_means) / train_stds

    all_test_set["Index"] = all_test_set["Index"].astype(int)
    return all_test_set

def test_single_houses(row, all_params, target_house):

    house_data = all_params["weights"][target_house]

    z = house_data["bias"]

    for j in range(len(all_params["best_features"])):
        feature_name = all_params["best_features"][j]
        z += house_data["weights"][j] * row[feature_name]

    res = 1.0 / (1.0 + math.exp(-z))
    return res

def test_all_houses(row, all_params):
    all_res = {}

    houses = all_params["weights"].keys()

    for _ in houses:
        all_res[_] = test_single_houses(row, all_params, _)

    return max(all_res, key=all_res.get)

def predict(all_params, cleand_test_data):
    result = {}

    for _, row in cleand_test_data.iterrows():
        student_id = row["Index"]
        result[student_id] = test_all_houses(row, all_params)

    return result

def save_in_csv(result, save_to_file):
    try:
        with open(save_to_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 1. header
            header = ["Index","Hogwarts House"]
            writer.writerow(header)

            # 2. ecrire les resultats
            for student_id, house_name in result.items():
                row = [student_id, house_name]
                writer.writerow(row)
        print(f"Resultats sauvegardés avec succès dans {save_to_file} !")

    except IOError as e:
        print(f"Erreur lors de l'écriture du CSV : {e}")
        exit(1)

# etapes:
# 1. lire model_params et stocker dans all_params_data pour filtrer les donnes dans test.csv
#    pour optimiser !!!=====================
# 2. lire dataset_test.csv avec best_features dans all_params et stocker dans DataFrame
# 3. calculer avec la formule(prediction), chaque fois calculer pour 4 houses
#  puis utiliser argmax(), stocker le resultat en format sujet dans un fichier houses.csv

def logreg_predict(test_set_file):
    file_model = "model_params.json"
    save_to_file = "houses.csv"

    all_params = load_model_params(file_model)
    cleand_test_data = load_test_set(all_params, test_set_file)
    result = predict(all_params, cleand_test_data)
    save_in_csv(result, save_to_file)

def main():
    args = sys.argv

    if len(args) != 2:
        print("Usage: ./logreg_predict.py datasets/dataset_test.csv")
        exit(1)

    # if (args[1] != "datasets/dataset_test.csv"):
    #     print("Usage: ./logreg_predict.py datasets/dataset_test.csv")
    #     exit(1)

    test_set_file = args[1]

    try:
        logreg_predict(test_set_file)

    except Exception as e:
        print(f"Erreur inattendue : {e}")
        exit(1)

if __name__ == "__main__":
    main()
