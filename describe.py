#!/usr/bin/env python3
import sys
import csv
import math

# fonction utils
# ici q est le poids de B (0.75)
# (1 - q) est le poids de A automatiquement(0.25)
# d0 = A * (1 - q)
# d1 = B * q
def get_percentile(sorted_list, q):
    # q peut etre 0.25 0.5 0.75 0 1
    n_total = len(sorted_list)

    # calculer la position
    k = (n_total - 1) * q
    f = math.floor(k)
    c = math.ceil(k)

    # si l'index est un entier
    if f == c:
        return float(sorted_list[int(k)])
    
    # sinon
    d0 = sorted_list[int(f)] * (c - k)
    d1 = sorted_list[int(c)] * (k - f)
    return float(d0 + d1)

# stocker dans le double dict !!!
def calculate_statistics(cleaned_dict):
    # un grand dict stocke resultat
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
        
        # min_nb = float(list_element[0])
        # for nb in list_element:
        #     if nb < min_nb:
        #         min_nb = nb
        
        # max_nb = float(list_element[0])
        # for nb in list_element:
        #     if nb > max_nb:
        #         max_nb = nb

        # sort pour min max 25% etc
        sorted_list = sorted(list_element)
        min_nb = float(sorted_list[0])
        max_nb = float(sorted_list[-1])

        q25 = get_percentile(sorted_list, 0.25)
        q50 = get_percentile(sorted_list, 0.5)
        q75 = get_percentile(sorted_list, 0.75)


        # stocker dans le petit dict pour cette statistique
        stats_result[cle]["Count"] = float(count)

        stats_result[cle]["Mean"] = mean

        stats_result[cle]["Std"] = std

        stats_result[cle]["Min"] = float(min_nb)

        stats_result[cle]["25%"] = q25

        stats_result[cle]["50%"] = q50

        stats_result[cle]["75%"] = q75
        
        stats_result[cle]["Max"] = float(max_nb)


        

    return stats_result

# --------------------------------------------------------------------------------
# etape 1. stocker tous les donnees dans dict_for_data
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

def display_statistics(stats_result):
    # header
    # header = "Feature".ljust(15) + "Count".rjust(10) + "Mean".rjust(12) + "Std".rjust(12) + "Min".rjust(12) + "25%".rjust(12) + "50%".rjust(12) + "75%".rjust(12) + "Max".rjust(12)
    # print(header)
    # print("-" * len(header))

    # 1. stocker tous les features(les clefs de cleaned_dict == stats_result)
    features = list(stats_result.key())

    # 2. definir la liste que je veux imprimer


    # chaque ligne

def describe(file_train):
    cleaned_dict = save_and_clean_data(file_train)
    stats_result = calculate_statistics(cleaned_dict)
    display_statistics(stats_result)
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
