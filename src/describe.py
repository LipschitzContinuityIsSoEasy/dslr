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

    missing_value_set = {"nan", "na", "null", "none", ""}

    # iterer un dict
    for cle, list_element in dict_for_data.items():

        # sauter index
        if cle == "Index":
            continue

        clean_list_for_cle = []
        is_column_valid = True

        for element in list_element:
            # 1. convertir en miniscule et enlever les espaces
            val_str = str(element).strip().lower()

            # 2. si c'est autorise
            if val_str in missing_value_set:
                continue
            # 3. essayer de convertir en chiffre
            try:
                nb = float(element)
                clean_list_for_cle.append(nb)
            except ValueError:
                # 4. si c'est pas autorise ou non chiffre
                is_column_valid = False
                break
        
        # panduan
        if is_column_valid and len(clean_list_for_cle) > 0:
            cleaned_dict[cle] = clean_list_for_cle

    return cleaned_dict

def save_and_clean_data(file_train):
    dict_for_data = save_all_data(file_train)
    cleaned_dict = clean_dict_for_data(dict_for_data)
    return cleaned_dict

# --------------------------------------------------------------------------------

def display_statistics(stats_result):
    # print(f"DEBUG: 我手里一共存了 {len(stats_result)} 个特征的统计信息")
    # # 打印所有特征的名字（大字典的键）
    # print("所有的 Feature 名字有：", list(stats_result.keys()))

    # first_feature_stats = list(stats_result.values())[0]
    # print(f"DEBUG: 它们的统计项包括: {list(first_feature_stats.keys())}")
    
    # first_cle = next(iter(stats_result))
    # print(f"第一个键是: {first_cle}")
    # print(f"它里面的内容是: {stats_result[first_cle]}")

    # header

    # 1. stocker tous les features(les clefs de cleaned_dict == stats_result)
    # comme `Arithmancy`, `Astronomy` ...
    features = list(stats_result.keys())

    # 2. stocker les indicateurs comme `Count`, `Mean` ...
    metrics = list(list(stats_result.values())[0].keys())

    # 3. imprimer header
    header = "".ljust(10)
    for _ in features:
        header += _[:10].rjust(12)
    print(header)
    print("-" * len(header))

    # 4. d'abord boucle exterieur(indicateurs)
    # ils decident au'on impriment combien de lignes
    for stat_name in metrics:

        # d'abord imprimer le nom de cet indicateur
        row_str = f"{stat_name:<12}"

        # puis boucle interieur(feature pour chaque indicateur)
        for feature_name in features:
            val = stats_result[feature_name][stat_name]

            # format float, 6 apres le virgule
            row_str += f"{val:>15.6f}" if isinstance(val, float) else f"{val:>15}"
        
        print(row_str)


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
