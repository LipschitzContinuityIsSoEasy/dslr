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
def load_test_set(all_params, test_file):
    try:
        all_test_set = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{test_file}' est introuvable.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Erreur : Le fichier '{test_file}' est vide.")
        exit(1)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV : {e}")
        exit(1)

    # j'ai juste besoin index et les cours, que des chiffres
    cleaned_test_set = [
        col for col in all_test_set.columns
        if pd.api.types.is_any_real_numeric_dtype(all_test_set[col])
    ]
    # remplir vide par mean(?)

# etapes:
# 1. lire model_params et stocker dans all_params_data pour filtrer les donnes dans test.csv
#    pour optimiser !!!=====================
# 2. lire dataset_test.csv avec best_features dans all_params et stocker dans DataFrame
# 3. calculer avec la formule(prediction), chaque fois calculer pour 4 houses
#  puis utiliser argmax(), stocker le resultat en format sujet dans un fichier houses.csv

def logreg_predict(test_file):
    file_model = "model_params.json"
    all_params = load_model_params(file_model)
    cleand_test_data = load_test_set(all_params, test_file)

def main():
    args = sys.argv

    if len(args) != 2:
        print("Usage: ./logreg_predict.py datasets/dataset_test.csv")
        exit(1)

    # if (args[1] != "datasets/dataset_test.csv"):
    #     print("Usage: ./logreg_predict.py datasets/dataset_test.csv")
    #     exit(1)

    file_name = args[1]

    try:
        logreg_predict(file_name)

    except Exception as e:
        print(f"Erreur inattendue : {e}")
        exit(1)

if __name__ == "__main__":
    main()
