#!/usr/bin/env python3
import sys
import csv

def clean_and_save_data(file_train):
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
        
        # imprimer cette dict (premiers 3)
        for k, v in list(dict_for_data.items())[:3]:
            print(f"抽屉（表头）: {k} -> 里面的数据列表: {v[:5]} ... (共 {len(v)} 条)")


    
    return dict_for_data

def describe(file_train):
    dict_for_data = clean_and_save_data(file_train)
    # stats_result = calculate_statistics(dict_for_data)
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
