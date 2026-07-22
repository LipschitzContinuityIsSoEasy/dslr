#!/usr/bin/env python3

import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def histogram(file_name):
    #  1. lire et stocker
    all_data = pd.read_csv(file_name)
    # print(all_data.head())

    # 2. trouver tous les cours, sans index, que des chiffres
    course_cols = [
        col for col in all_data.columns
        if col != 'Index' and pd.api.types.is_any_real_numeric_dtype(all_data[col])
    ]

    # creer les repertoires!
    os.makedirs("visualisations/hist", exist_ok=True)

    # 3. boucle pour chaque cours et dessiner
    for col in course_cols:
        # 1. creer un canva
        plt.figure(figsize=(10, 6))

        # 2. utiliser histplot dans Seaborn
        # x est cours, hue est nom de house
        # multiple="layer": plusieurs couches
        # alpha: transparent 0.5
        sns.histplot(
            data = all_data,
            x = col,
            hue = "Hogwarts House",
            multiple = "layer",
            alpha = 0.5,
            kde = True  # optionel
        )

        # 3. ajouter nom et label
        plt.title(f"Histogram of {col} by Hogwarts House", fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Count", fontsize=12)

        # 4. sauvegarder en images
        output_filename = f"visualisations/hist/{col.replace(' ', '_')}_hist.png"
        plt.savefig(output_filename)

        # close et free, dessiner le cours suivant
        plt.close()
        print(f"Sauvegarde: {output_filename}")

def main():
    args = sys.argv
    if len(args) < 2:
        print("Usage: ./histogram.py <dataset.csv>")
        exit(1)

    file_name = args[1]

    try:
        histogram(file_name)

    except Exception as e:
        print(f"Erreur inattendue : {e}")
        exit(1)

if __name__ == "__main__":
    main()
