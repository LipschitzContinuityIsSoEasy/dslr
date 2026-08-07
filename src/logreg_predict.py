#!/usr/bin/env python3

import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import csv

from sklearn.feature_selection import f_classif

# juste lire model pramametresjson 1 fois et stocker dans un data
# etape:
# 1. lire test.csv et stocker 
def load_test_set(filename):
    try:
        all_test_set = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{filename}' est introuvable.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Erreur : Le fichier '{filename}' est vide.")
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
# 1. lire dataset_test.csv et stocker dans DataFrame
# 2. lire weights.csv et stocker dans weights_data
# 3. calculer avec la formule(prediction), chaque fois calculer pour 4 houses
#  puis utiliser argmax(), stocker le resultat en format sujet dans un fichier houses.csv

def logreg_predict(filename):
    test_data = load_test_set(filename)

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
