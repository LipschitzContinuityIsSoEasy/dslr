#!/usr/bin/env python3

"""Descriptive statistics for every numeric column of a dataset.

Reads a CSV file, keeps only its numeric columns, and prints an
aligned table of count, mean, standard deviation, quartiles, range,
interquartile range, and skewness.
"""

import sys
import csv
import math

from utils import short_name


# fonction utils
# ici q est le poids de B (0.75)
# (1 - q) est le poids de A automatiquement(0.25)
# d0 = A * (1 - q)
# d1 = B * q
def get_percentile(sorted_list: list[float], q: float) -> float:
    """Compute a percentile with linear interpolation.

    Args:
        sorted_list: Values of one column, sorted in ascending order.
        q: Quantile to compute, between 0 and 1.

    Returns:
        The value at the requested quantile.
    """
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


def get_skewness(list_element: list[float], mean: float) -> float:
    """Compute the adjusted skewness of one column.

    Args:
        list_element: Values of one column.
        mean: Mean of the column.

    Returns:
        Skewness of the column, 0.0 when it cannot be computed.
    """
    count = len(list_element)

    # at least 3 values are required
    if count < 3:
        return 0.0

    m2 = sum((x - mean) ** 2 for x in list_element) / count
    m3 = sum((x - mean) ** 3 for x in list_element) / count

    # if all values are identical - there's nothing to measure
    if m2 == 0.0:
        return 0.0

    # sample size correction
    correction = math.sqrt(count * (count - 1)) / (count - 2)

    return correction * m3 / m2 ** 1.5


# stocker dans le double dict !!!
def calculate_statistics(
    cleaned_dict: dict[str, list[float]],
    missing_dict: dict[str, int]
) -> dict[str, dict[str, float]]:
    """Compute every metric of every numeric column.

    Args:
        cleaned_dict: Numeric values of each column.
        missing_dict: Number of missing values of each column.

    Returns:
        Metrics of each column, keyed by column name then metric name.
    """
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

        # asymmetry of the distribution
        skew = get_skewness(list_element, mean)

        # stocker dans le petit dict pour cette statistique
        stats_result[cle]["Count"] = float(count)

        # how many were removed by the cleanup
        stats_result[cle]["Missing"] = float(missing_dict[cle])

        stats_result[cle]["Mean"] = mean

        stats_result[cle]["Std"] = std

        stats_result[cle]["Min"] = float(min_nb)

        stats_result[cle]["25%"] = q25

        stats_result[cle]["50%"] = q50

        stats_result[cle]["75%"] = q75

        stats_result[cle]["Max"] = float(max_nb)

        stats_result[cle]["Range"] = float(max_nb - min_nb)

        # Interquartile range
        stats_result[cle]["IQR"] = q75 - q25

        stats_result[cle]["Skew"] = skew

    return stats_result


# ---------------------------------------------------------------------------
# etape 1. stocker tous les donnees dans dict_for_data
def save_all_data(file_train: str) -> dict[str, list[str]]:
    """Read a CSV file into one list of raw values per column.

    Args:
        file_train: Path to the CSV file.

    Returns:
        Mapping of each header name to its raw values.

    Raises:
        ValueError: If the file cannot be read or is empty.
    """
    dict_for_data = {}

    try:
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
    except FileNotFoundError:
        raise ValueError(f"file '{file_train}' is not found.")
    except StopIteration:
        raise ValueError(f"file '{file_train}' is empty.")
    except Exception as e:
        raise ValueError(f"cannot read '{file_train}': {e}")

    return dict_for_data


# etape 2. iterer cette dict, nettoyer et remplacer
def clean_dict_for_data(
    dict_for_data: dict[str, list[str]]
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Keep the numeric columns and count their missing values.

    Args:
        dict_for_data: Raw values of every column of the dataset.

    Returns:
        Numeric values of each kept column, and how many values were
        missing in it.
    """
    cleaned_dict = {}
    missing_dict = {}

    missing_value_set = {"nan", "na", "null", "none", ""}

    # iterer un dict
    for cle, list_element in dict_for_data.items():

        # sauter index
        if cle == "Index":
            continue

        clean_list_for_cle = []
        missing_count = 0
        is_column_valid = True

        for element in list_element:
            # 1. convertir en miniscule et enlever les espaces
            val_str = str(element).strip().lower()

            # 2. si c'est autorise
            if val_str in missing_value_set:
                # count for the "Missing" statistic
                missing_count += 1
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
            missing_dict[cle] = missing_count

    return cleaned_dict, missing_dict


def save_and_clean_data(
    file_train: str
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Read the dataset and keep only its numeric columns.

    Args:
        file_train: Path to the CSV file.

    Returns:
        Numeric values of each kept column, and how many values were
        missing in it.
    """
    dict_for_data = save_all_data(file_train)
    cleaned_dict, missing_dict = clean_dict_for_data(dict_for_data)
    return cleaned_dict, missing_dict


# ---------------------------------------------------------------------------

# ici il faut stocker dans un fichier csv ou json, j'ai choisi json pour la
# suite
def display_statistics(stats_result: dict[str, dict[str, float]]) -> None:
    """Print the statistics as an aligned table.

    Args:
        stats_result: Metrics of each numeric column.

    Returns:
        None.
    """
    # print(f"DEBUG: 我手里一共存了 {len(stats_result)} 个特征的统计信息")
    # # 打印所有特征的名字（大字典的键）
    # print("所有的 Feature 名字有：", list(stats_result.keys()))

    # first_feature_stats = list(stats_result.values())[0]
    # print(f"DEBUG: 它们的统计项包括: {list(first_feature_stats.keys())}")

    # first_cle = next(iter(stats_result))
    # print(f"第一个键是: {first_cle}")
    # print(f"它里面的内容是: {stats_result[first_cle]}")

    # pas besoin ... ========================================================
    # ouvrir un json
    # save_in_file = "model_params.json"
    # three_dimention_dict = {
    #     "statistics": stats_result
    # }

    # try:
    #     with open(save_in_file, "w") as datafile:
    #         json.dump(three_dimention_dict, datafile, indent=4)
    # except PermissionError:
    #     print(f"Erreur : Le fichier '{save_in_file}' permission denied")
    #     exit(1)
    # except Exception as e:
    #     print(f"Erreur lors de l'enregistrement du fichier JSON : {e}")
    #     exit(1)

    # 1. stocker tous les features(les clefs de cleaned_dict == stats_result)
    # comme `Arithmancy`, `Astronomy` ...
    features = list(stats_result.keys())

    # 2. stocker les indicateurs comme `Count`, `Mean` ...
    metrics = list(list(stats_result.values())[0].keys())

    # 3. imprimer header
    # same width as the lines below, otherwise no alignment
    header = "".ljust(12)
    for _ in features:
        header += short_name(_).rjust(16)
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
            row_str += (
                f"{val:>16.6f}" if isinstance(val, float)
                else f"{val:>16}"
            )

        print(row_str)


def describe(file_train: str) -> None:
    """Display the statistics of every numeric column of a dataset.

    Args:
        file_train: Path to the CSV file.

    Returns:
        None.

    Raises:
        ValueError: If the dataset has no numeric column.
    """
    cleaned_dict, missing_dict = save_and_clean_data(file_train)

    if not cleaned_dict:
        raise ValueError("dataset contains no numeric column.")

    stats_result = calculate_statistics(cleaned_dict, missing_dict)
    display_statistics(stats_result)


def main() -> None:
    """Describe each numeric feature of the dataset.

    Expects the dataset path as an argument:
    ./describe.py <dataset.csv>

    Returns:
        None.
    """
    # etape: 1. creer un dictionaire
    # {"colon1": [chiffre_1, chiffre_2], ..., "colon2":[chiffre1, ...], ...}
    args = sys.argv

    #  ??trop tard?
    if len(args) < 2:
        print("Usage: ./describe.py <dataset.csv>")
        exit(1)

    file_train = args[1]

    try:
        describe(file_train)
    except Exception as e:
        print("Error: " + e.args[0] if e.args else "unexpected error.")
        exit(1)


if __name__ == "__main__":
    main()
