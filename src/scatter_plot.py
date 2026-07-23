#!/usr/bin/env python3

import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def scatter_plot(file_name):
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

    numeric_data = all_data[course_cols]

    # pour tout afficher
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 1000)
    # pd.set_option('display.precision', 4)
    # print(numeric_data.corr())

    corr_matrix = numeric_data.corr()

    # aplatir la matrice et sort
    stacked_corr = corr_matrix.stack().sort_values(ascending=False)

    filtered_corr = stacked_corr[stacked_corr.index.get_level_values(0) != stacked_corr.index.get_level_values(1)]

    # print(" --- les premiers 4 corr--- ")

    # print(filtered_corr.head(4))

    best_pair = filtered_corr.index[0]
    course1, course2 = best_pair

    print(f"Les deux cours sont: {course1} et {course2}")

    # creer les repertoires!
    # chemin absolu
    # os.makedirs("outputs/figures", exist_ok=True)
    # obtenir lui-meme d'abord
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    output_dir = os.path.join(project_root, "outputs", "plot")
    os.makedirs(output_dir, exist_ok=True)

    # creer le canva
    plt.figure(figsize=(8,6))

    # dessiner
    sns.scatterplot(
        data=all_data,
        x=course1,
        y=course2,
        hue="Hogwarts House", # optionnel
        alpha=0.7
    )

    plt.title(f"Scatter Plot between {course1} and {course2}")
    plt.xlabel(course1)
    plt.ylabel(course2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    output_path = os.path.join(output_dir, "scatter_plot.png")
    plt.savefig(output_path)
    print(f"Sauvegarde: {output_path}")

    # plt.show()

def main():
    args = sys.argv
    if len(args) < 2:
        print("Usage: ./scatter_plot.py <dataset.csv>")
        exit(1)

    file_name = args[1]

    try:
        scatter_plot(file_name)

    except Exception as e:
        print(f"Erreur inattendue : {e}")
        exit(1)

if __name__ == "__main__":
    main()
