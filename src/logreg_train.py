#!/usr/bin/env python3

import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_classif

def load_and_preprocess_data(file_name):
    #  1. lire et stocker
    try:
        # abs_file_path = os.path.abspath(file_name)
        all_data = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_name}' est introuvable.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Erreur : Le fichier '{file_name}' est vide.")
        exit(1)
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier CSV : {e}")
        exit(1)

    # 1.1 verifier si House exist
    if "Hogwarts House" not in all_data.columns:
        print("Erreur : La colonne 'Hogwarts House' est introuvable dans le dataset.")
        exit(1)

    # 2. trouver tous les cours, sans index, que des chiffres
    course_cols = [
        col for col in all_data.columns
        if col != 'Index' and pd.api.types.is_any_real_numeric_dtype(all_data[col])
    ]

    if not course_cols:
        print("Avertissement : Aucune colonne numérique valide n'a été trouvée pour les cours.")
        return

    # netoyyer tous les NaN
    clean_data = all_data[course_cols + ['Hogwarts House']].dropna()

    X = clean_data[course_cols]
    y = clean_data['Hogwarts House']

    # 4. Calculer la valeur F (F-score) et la valeur P pour chaque cours
    f_values, p_values = f_classif(X, y)

    # 5. Regrouper dans un DataFrame et trier par valeur F décroissante
    feature_ranking = pd.DataFrame({
        'Feature': course_cols,
        'F_Value': f_values,
        'P_Value': p_values
    }).sort_values(by='F_Value', ascending=False).reset_index(drop=True)

    print("--- Classement des cours selon leur capacité à différencier les maisons (plus le F-score est élevé, meilleur est le résultat de classification) ---")
    print(feature_ranking.to_string(index=False))

    # 6. Sélectionner automatiquement les 10 meilleures caractéristiques avec le F-score le plus élevé pour les utiliser dans la régression logistique
    top_k = 10
    best_features = feature_ranking['Feature'].head(top_k).tolist()
    print(f"\nCaractéristiques recommandées pour la régression logistique (top {top_k}) :")
    print(best_features)

    # return donnee: juste retouner les cols 'Hogwarts House' et les premiers 10 cols
    return clean_data[best_features + ['Hogwarts House']], best_features

def logreg_train(file_name):
    # 1. faire 1 et 2
    cleaned_data = load_and_preprocess_data(file_name)

    # 2. faire 3 normalisation
    # normalized_data = normalize_data(cleaned_data)

    # 3. boucle et aussi stocker dans un fichier a la fin
    # train_all_houses(normalized_data)

# 1. lire les donnee, stocker dans DataFrame dans Pandas, et nettoyer
# 2. F-score, choisir les premieres 10
# 3. normaliser tous ces premieres 10 caracteristiques
# 4. une boucle pour les 4 maisons
#     chaque fois est 1, les autres sont 0, faire 30 000 iterations
#     obtenir 10 w et 1 biais
#     je definit hyperparametres Learning-rate et nb d'iterations
# 5. stocker dans un fichier
def main():
    args = sys.argv

    if len(args) != 2:
        print("Usage: ./logreg_train.py datasets/dataset_train.csv")
        exit(1)

    # if (args[1] != "datasets/dataset_train.csv"):
    #     print("Usage: ./logreg_train.py datasets/dataset_train.csv")
    #     exit(1)

    file_name = args[1]

    try:
        logreg_train(file_name)

    except Exception as e:
        print(f"Erreur inattendue : {e}")
        exit(1)

if __name__ == "__main__":
    main()
