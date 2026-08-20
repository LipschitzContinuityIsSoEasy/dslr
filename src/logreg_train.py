#!/usr/bin/env python3

import sys
import pandas as pd
import math
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

def calculate_single_prediction_and_error(row, best_features, weights, bias, target_house):
    num_features = len(best_features)

    # 1. calculer score lineaire z = b + w1*x1 + w2*x2 + ..
    z = bias
    for j in range(num_features):
        feature_name = best_features[j]
        z += weights[j] * row[feature_name]

    # 2. protection
    if z < -700:
        prediction = 0.0
    elif z > 700:
        prediction = 1.0
    else:
        prediction = 1.0 / (1.0 + math.exp(-z))

    # 3. real lable
    real_house = row['Hogwarts House']
    y = 0.0
    if real_house == target_house:
        y = 1.0

    # 4. calculer l'error
    error = prediction - y
    return error

def train_single_house(pipeline_data, target_house):
    option = pipeline_data["option"]
    normalized_data = pipeline_data["normalized_data"]
    best_features = pipeline_data["best_features"]
    learning_rate = pipeline_data["learning_rate"]
    epochs = pipeline_data["epochs"]
    batch_size = pipeline_data["batch_size"]
    
    m = len(normalized_data)
    num_features = len(best_features)
    weights = [0.0] * num_features
    bias = 0.0

    if option == "BGD":
        batch_size = m
        shuffle = False
    elif option == "SGD":
        batch_size = 1
        shuffle = True
    elif option == "minibatch":
        batch_size = 32
        shuffle = True
    elif option == "minibatch-noshuffle":
        batch_size = 32
        shuffle = False
    else:
        raise ValueError(f"Option inconnue : {option}")

    for _ in range(epochs):
        if shuffle:
            current_data = normalized_data.sample(frac=1).reset_index(drop=True)
        else:
            current_data = normalized_data

        # # stocker l'érreur carrée totale pour cet itération
        # current_sum_mse = 0.0

        for i in range(0, m, batch_size):
            batch = current_data.iloc[i : i + batch_size]
            current_batch_size = len(batch)

            sum_error_weights = [0.0] * num_features
            sum_error_bias = 0.0

            for _, row in batch.iterrows():
                error = calculate_single_prediction_and_error(row, best_features, weights, bias, target_house)

                sum_error_bias += error
                for j in range(num_features):
                    feature_name = best_features[j]
                    sum_error_weights[j] += error * row[feature_name]
            
            bias = bias - (learning_rate * (1 / current_batch_size) * sum_error_bias)
            for j in range(num_features):
                weights[j] = weights[j] - (learning_rate * (1 / current_batch_size) * sum_error_weights[j])

    return weights, bias

def save_all_to_json(all_model_data, filename_json):
    """Save all trained model parameters (features, means, stds, weights, biases) 
    into a JSON file with proper error handling.
    """
    try:
        # Serialize and write model data to JSON
        with open(filename_json, "w") as f:
            json.dump(all_model_data, f, indent=4)
            print(f"Complete model successfully saved to '{filename_json}'!")
    except FileNotFoundError:       
        print(f"Error: The file '{filename_json}' was not found.")
        exit(1)
    except PermissionError:
        print(f"Error: Permission denied for file '{filename_json}'.")
        exit(1)
    except Exception as e:
        print(f"I/O error occurred while writing to JSON: {e}")
        exit(1)
    except IOError as e:
        print(f"Unexpected error while saving JSON: {e}")
        exit(1)
    
def train_all_houses(pipeline_data):
    """Train One-vs-All (OvR) binary classifiers for all 4 Hogwarts houses 
    and save the combined model parameters to a JSON file.
    """
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    filename_json="model_params.json"

    best_features = pipeline_data["best_features"]
    train_means = pipeline_data["train_means"]
    train_stds = pipeline_data["train_stds"]

    weights_bias = {}

    for house in houses:
        print(f"Training model for: {house}...")
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
    # Save all trained parameters to JSON
    save_all_to_json(all_model_data, filename_json)
    print("All models trained successfully!")

def logreg_train(file_name, option):
    """Execute the full logistic regression training pipeline:
        
    1. Load and clean the dataset.
    2. Normalize features using Z-score standardization: (x - mean) / std.
    3. Train One-vs-All (OvR) binary classifiers for the 4 Hogwarts houses.
    4. Save trained weights and metadata to a JSON file.
    """    
    pipeline_data = {
        "option": option,
        "learning_rate": 0.1,
        "epochs" : 300,
        "batch_size" : 32
    }
    # 1. Load and preprocess data
    pipeline_data = load_and_preprocess_data(file_name, pipeline_data)

    # 2. Normalize features (Z-score)
    pipeline_data = normalize_data(pipeline_data)

    # 3. Train all houses and save model parameters
    train_all_houses(pipeline_data)

def main() -> None:
    """Train a logistic regression model using specified gradient descent optimizer.

    Expects the dataset path and an optional optimizer as arguments:
    ./logreg_train.py <dataset_path.csv> [BGD/SGD/minibatch/minibatch-noshuffle]

    Returns:
        None.
    """
    args = sys.argv

    if len(args) < 2:
        print("Usage: ./logreg_train.py datasets/dataset_train.csv [option]")
        print("Optimizers available: BGD (default), SGD, minibatch, minibatch-noshuffle")
        exit(1)

    file_name = args[1]

    if len(args) > 2:
        option = args[2]
    else:
        option = "BGD"

    try:
        logreg_train(file_name, option)

    except Exception as e:
        print("Error: " + e.args[0] if e.args else "unexpected error.")
        exit(1)

if __name__ == "__main__":
    main()
