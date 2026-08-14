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

def load_and_preprocess_data(file_name, pipeline_data):
    #  1. lire et stocker
    try:
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

    # 0.1 d'abord calculer mean
    raw_means = {}

    for col in course_cols:
        mean_val = all_data[col].mean()
        raw_means[col] = mean_val
        # 0.2 remplire avec 
        all_data[col] = all_data[col].fillna(mean_val)

    # non : netoyyer tous les NaN/None
    clean_data = all_data[course_cols + ['Hogwarts House']].dropna(subset=['Hogwarts House'])

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

    # ===========================================================================================================================================================================
    # ici calculer stds
    train_stds = {}
    train_means = {col: raw_means[col] for col in best_features}

    for col in best_features:
        std_val = clean_data[col].std()
        if std_val == 0:
            std_val = 1.0
        train_stds[col] = std_val

    pipeline_data.update({
        "cleaned_data": clean_data[best_features + ['Hogwarts House']],
        "best_features": best_features,
        "train_means": train_means,
        "train_stds": train_stds
    })

    return pipeline_data

def normalize_data(pipeline_data):
    cleaned_data = pipeline_data["cleaned_data"]
    best_features = pipeline_data["best_features"]
    train_means = pipeline_data["train_means"]
    train_stds = pipeline_data["train_stds"]

    # 1. copier et coller
    normalized_copie = cleaned_data.copy()

    # 2. supprimer les houses
    houses = normalized_copie.pop('Hogwarts House')

    # 3. juste les premiere 10 caracteristiques
    for col in best_features:
        mean_val = train_means[col]
        std_val = train_stds[col]
        normalized_copie[col] = (normalized_copie[col] - mean_val) / std_val

    # 4. rajouter les houses
    normalized_copie['Hogwarts House'] = houses

    # remplacer cleaned_data par normalized_copie
    pipeline_data.pop("cleaned_data", None)
    pipeline_data["normalized_data"] = normalized_copie

    return pipeline_data


# BGD
def bgd(pipeline_data, target_house):
    print("BGD")
    normalized_data = pipeline_data["normalized_data"]
    best_features = pipeline_data["best_features"]
    learning_rate = pipeline_data["learning_rate"]
    epochs = pipeline_data["epochs"]
    
    # 1. ici on utilsie batch gradient descent(BGD)
    m = len(normalized_data)
    num_features = len(best_features)

    # initialiser tous en 0
    weights = [0.0] * num_features
    bias = 0.0

    for _ in range(epochs):
        # Batch Gradient Descent (BGD)
        # La Descente de Gradient par Lot

        # initialiser les accumulateurs
        sum_error_weights = [0.0] * num_features
        sum_error_bias = 0.0

        # # stocker l'érreur carrée totale pour cet itération
        # current_sum_mse = 0.0

        for i in range(m):
            # 0. obtenir les donnees d'un eleve en i-eme ligne
            row = normalized_data.iloc[i]

            # 1. calculer score lineaire z = b + w1*x1 + w2*x2 + ..
            z = bias
            for j in range(num_features):
                feature_name = best_features[j]
                z += weights[j] * row[feature_name]

            # 2. remplacer dans Sigmoid pour avoir le resultat
            prediction = 1.0 / (1.0 + math.exp(-z))

            # 3. savoir le vrai label y pour l'eleve actuel (0 ou 1)
            real_house = row['Hogwarts House']
            y = 0.0
            if (real_house == target_house):
                y = 1.0

            # 4. calculer l'error
            error = prediction - y

            # 5. accumuler les gradients
            sum_error_bias += error
            for j in range(num_features):
                feature_name = best_features[j]
                sum_error_weights[j] += error * row[feature_name]

        # 6. apres une boucle, renouveler tout en meme temps
        bias = bias - (learning_rate * (1/m) * sum_error_bias)
        for j in range(num_features):
            weights[j] = weights[j] - (learning_rate * (1 / m) * sum_error_weights[j])

    return weights, bias


# SGD
def sgd(pipeline_data, target_house):
    print("SGD")
    normalized_data = pipeline_data["normalized_data"]
    best_features = pipeline_data["best_features"]
    learning_rate = pipeline_data["learning_rate"]
    epochs = pipeline_data["epochs"]
    
    # 1. ici on utilsie batch gradient descent(BGD)
    m = len(normalized_data)
    num_features = len(best_features)

    # initialiser tous en 0
    weights = [0.0] * num_features
    bias = 0.0

    for _ in range(epochs):
        # stochastic gradient descent (SGD)

        # aleatoire pour eviter Data Ordering Bias
        shuffled_data = normalized_data.sample(frac=1).reset_index(drop =True)

        # pas besoin de sum_error

        for i in range(m):
            # 0. obtenir les donnees d'un eleve en i-eme ligne
            row = shuffled_data.iloc[i]

            # 1. calculer score lineaire z = b + w1*x1 + w2*x2 + ..
            z = bias
            for j in range(num_features):
                feature_name = best_features[j]
                z += weights[j] * row[feature_name]

            # 2. remplacer dans Sigmoid pour avoir le resultat
            prediction = 1.0 / (1.0 + math.exp(-z))

            # 3. savoir le vrai label y pour l'eleve actuel (0 ou 1)
            real_house = row['Hogwarts House']
            y = 0.0
            if (real_house == target_house):
                y = 1.0

            # 4. calculer l'error
            error = prediction - y

            # 5. pas besoin d'accumuler
            bias = bias - learning_rate * error
            for j in range(num_features):
                feature_name = best_features[j]
                weights[j] = weights[j] - learning_rate * error * row[feature_name]

    return weights, bias

def train_single_house(pipeline_data, target_house):
    option = pipeline_data["option"]
    if option == "BGD":
        return bgd(pipeline_data, target_house)
    elif option == "SGD":
        return sgd(pipeline_data, target_house)
    elif option == "minibatch":
        pass
        # return minibatch(pipeline_data, target_house)
    else:
        raise ValueError(f"Option inconnue : {option}")

def save_weights_to_csv(all_parameters, filename):
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 1. headers (House, Bias, W0, W1, ... W9)
            first_house_data = next(iter(all_parameters.values()))
            num_features = len(first_house_data['weights'])
            header = ['House', 'Bias'] + [f'W_{i}' for i in range(num_features)]
            writer.writerow(header)
            
            # 2. ecrire les donnees dans le fichier
            for house, params in all_parameters.items():
                row = [house, params['bias']] + params['weights']
                writer.writerow(row)
                
        print(f"Poids sauvegardés avec succès dans {filename} !")

    except IOError as e:
        print(f"Erreur lors de l'écriture du CSV : {e}")
        exit(1)

def save_all_to_json(all_model_data, filename_json):
    try:
        # d'abord stocker les donnees dans le fichier
        with open(filename_json, "w") as f:
            json.dump(all_model_data, f, indent=4)
            print(f"Modèle complet (features, means, weights) sauvegardé avec succès dans '{filename_json}' !")
    except FileNotFoundError:       
        print(f"Erreur : Le fichier '{filename_json}' not found")
        exit(1)
    except PermissionError:
        print(f"Erreur : Le fichier '{filename_json}' permission denied")
        exit(1)
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du fichier JSON : {e}")
        exit(1)
    except IOError as e:
        print(f"Erreur lors de l'écriture du CSV : {e}")
        exit(1)
    
# OvR (One-vs-All)
def train_all_houses(pipeline_data):
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    # filename="weights.csv"
    filename_json="model_params.json"

    best_features = pipeline_data["best_features"]
    train_means = pipeline_data["train_means"]
    train_stds = pipeline_data["train_stds"]

    weights_bias = {}

    for house in houses:
        print(f"Entraînement du modèle pour : {house}...")
        weights, bias = train_single_house(pipeline_data, house)

        weights_dict = {}

        for j in range(len(best_features)):
            feature_name = best_features[j]
            weight_value = weights[j]

            weights_dict[feature_name] = weight_value

        weights_bias[house] = {
            "weights" : weights_dict,
            "bias" : bias
        }

    all_model_data = {
        "best_features" : best_features,
        "train_means" : train_means,
        "train_stds" : train_stds,
        "weights" : weights_bias
    }
    # ecrire dans un json
    save_all_to_json(all_model_data, filename_json)
    print("Tous les modèles sont entraînés avec succès !")

def logreg_train(file_name, option):
    pipeline_data = {
        "option": option,
        "learning_rate": 0.1,
        "epochs" : 300,
        "batch_size" : 32
    }
    # 1. faire 1 et 2
    pipeline_data = load_and_preprocess_data(file_name, pipeline_data)

    # 2. faire 3 normalisation
    # =====================================================================================================================================================
    # attention: ici oall_model_datan utilise Z-score(x-niu)/sigma, pas (x-min)/(max-min)
    pipeline_data = normalize_data(pipeline_data)

    # 3. boucle et aussi stocker dans un fichier a la fin
    train_all_houses(pipeline_data)

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

    if len(args) < 2:
        print("Usage: ./logreg_train.py datasets/dataset_train.csv [option]")
        print("Optimizers available: BGD (default), SGD, minibatch")
        exit(1)

    # if (args[1] != "datasets/dataset_train.csv"):
    #     print("Usage: ./logreg_train.py datasets/dataset_train.csv")
    #     exit(1)

    file_name = args[1]

    if len(args) > 2:
        option = args[2]
    else:
        option = "BGD"

    try:
        logreg_train(file_name, option)

    except Exception as e:
        print(f"Erreur inattendue : {e}")
        exit(1)

if __name__ == "__main__":
    main()
