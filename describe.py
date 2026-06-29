#!/usr/bin/env python3
import sys
import csv
import math

# stocker dans le double dict !!!
def calculate_statistics(cleaned_dict):
    # il y a petit dict pour 8 elements
    stats_result = {}

    #  attention, iteration dans cleaned_dict
    for cle, list_element in cleaned_dict.items():
        # initialiser stats_result avec les cles de cleaned_dict
        stats_result[cle] = {}

        # calculer et stocker
        count = len(list_element)
        
        mean = 0.0
        if count > 0:
            mean = sum(list_element) / float(count)
        
        std = 0.0
        if count > 1:
            variance_sum = sum((x - mean) ** 2 for x in list_element)

            variance = variance_sum / (count - 1)

            std = math.sqrt(variance)
        
        min_nb = float(list_element[0])
        for nb in list_element:
            if nb < min_nb:
                min_nb = nb
        
        max_nb = float(list_element[0])
        for nb in list_element:
            if nb > max_nb:
                max_nb = nb


        # stocker dans le petit dict pour cette statistique
        stats_result[cle]["Count"] = float(count)

        stats_result[cle]["Mean"] = mean

        stats_result[cle]["Std"] = std

        stats_result[cle]["Min"] = float(min_nb)
        
        stats_result[cle]["Max"] = float(max_nb)


        

    return stats_result

# --------------------------------------------------------------------------------
# etape 1. stocker tous les donnees das dict_for_data
def save_all_data(file_train):
    dict_for_data = {}

    with open(file_train, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        
        # 1. obtenir le header
        header = next(reader)

        # 2. initialiser la clef, chaque header a une liste vide []
        for col in header:
            dict_for_data[col] = []
        
        # 3. boucle lire donnee
        for row in reader:
            # chaque row est une liste
            # 0-eme correspond a 0-eme col
            # n-eme correspond a n-eme col
            for index, element in enumerate(row):
                # trouver le col correspond par
                # index de row et index de header en meme teps
                header_name = header[index]
                dict_for_data[header_name].append(element)

    return dict_for_data

# etape 2. iterer cette dict, nettoyer et remplacer
def clean_dict_for_data(dict_for_data):
    cleaned_dict = {}

    # iterer un dict
    for cle, list_element in dict_for_data.items():

        # sauter index
        if cle == "Index":
            continue

        clean_list_for_cle = []

        for element in list_element:
            if element == "":
                continue
            try:
                nb = float(element)
                clean_list_for_cle.append(nb)
            except ValueError:
                pass
        
        # panduan
        if len(clean_list_for_cle) > 0:
            cleaned_dict[cle] = clean_list_for_cle

    return cleaned_dict

def save_and_clean_data(file_train):
    dict_for_data = save_all_data(file_train)
    cleaned_dict = clean_dict_for_data(dict_for_data)
    return cleaned_dict

# --------------------------------------------------------------------------------

def describe(file_train):
    cleaned_dict = save_and_clean_data(file_train)
    stats_result = calculate_statistics(cleaned_dict)
    # display_statistics(stats_result)
    pass

def main():
# etape: 1. creer un dictionaire {"colon1": [chiffre_1, chiffre_2], ..., "colon2":[chiffre1, chiffre2...], ...}
    args = sys.argv

    #  ??trop tard?
    if len(args) < 2:
        print("Usage: ./describe.py <dataset.csv>")
        exit(1)

    file_train = args[1]
    try:
        describe(file_train)
            
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        exit(1)


if __name__ == "__main__":
    main()
