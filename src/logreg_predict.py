#!/usr/bin/env python3

import sys
import pandas as pd
import math
import csv
import json

def load_model_params(file_model: str) -> dict:
    """Load trained model parameters from a JSON file.

    Args:
        file_model: Path to the JSON file containing the model parameters.

    Returns:
        A dictionary containing the trained model parameters.
    """
    try:
        with open(file_model, "r") as f:
            all_params = json.load(f)

    except FileNotFoundError:
        print(f"Error: The file '{file_model}' was not found.")
        exit(1)

    except PermissionError:
        print(f"Error: Permission denied for '{file_model}'.")
        exit(1)

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{file_model}': {e}")
        exit(1)

    except IOError as e:
        print(f"Error while reading '{file_model}': {e}")
        exit(1)

    return all_params

def load_test_set(
    all_params: dict,
    test_file: str
) -> pd.DataFrame:
    """Load and normalize the test dataset.

    Args:
        all_params: Trained model parameters.
        test_file: Path to the test CSV file.

    Returns:
        The cleaned and normalized test dataset.
    """
    try:
        all_test_set = pd.read_csv(
            test_file,
            usecols=["Index"] + all_params["best_features"]
        )

    except FileNotFoundError:
        print(f"Error: The file '{test_file}' was not found.")
        exit(1)

    except pd.errors.EmptyDataError:
        print(f"Error: The file '{test_file}' is empty.")
        exit(1)

    except Exception as e:
        print(f"Error while reading the CSV file: {e}")
        exit(1)

    if all_test_set["Index"].isnull().any():
        print(
            "Warning: Missing values found in the Index column."
        )

    for col in all_params["best_features"]:
        train_mean = all_params["train_means"][col]
        train_std = all_params["train_stds"][col]

        if train_std == 0:
            train_std = 1.0

        all_test_set[col] = all_test_set[col].fillna(
            train_mean
        )

        all_test_set[col] = (
            all_test_set[col] - train_mean
        ) / train_std

    all_test_set["Index"] = all_test_set["Index"].astype(int)

    return all_test_set

def test_single_house(
    row: pd.Series,
    all_params: dict,
    target_house: str
) -> float:
    """Calculate the prediction probability for one Hogwarts house.

    Args:
        row: Student data used for prediction.
        all_params: Trained model parameters.
        target_house: Hogwarts house to predict.

    Returns:
        The prediction probability for the target house.
    """
    house_data = all_params["weights"][target_house]

    z = house_data["bias"]

    for feature_name in all_params["best_features"]:
        weight = house_data["weights"][feature_name]

        z += weight * row[feature_name]

    if z < -700:
        res = 0.0

    elif z > 700:
        res = 1.0

    else:
        res = 1.0 / (1.0 + math.exp(-z))

    return res


def test_all_houses(
    row: pd.Series,
    all_params: dict
) -> str:
    """Predict the house with the highest probability.

    Args:
        row: Student data used for prediction.
        all_params: Trained model parameters for all houses.

    Returns:
        The name of the predicted Hogwarts house.
    """
    all_res = {}

    houses = all_params["weights"].keys()

    for house in houses:
        all_res[house] = test_single_house(
            row,
            all_params,
            house
        )

    return max(all_res, key=all_res.get)


def predict(
    all_params: dict,
    cleaned_test_data: pd.DataFrame
) -> dict:
    """Predict the Hogwarts house for each student.

    Args:
        all_params: Trained model parameters.
        cleaned_test_data: Cleaned test dataset.

    Returns:
        A dictionary containing each student index and predicted house.
    """
    result = {}

    for _, row in cleaned_test_data.iterrows():
        student_id = row["Index"]

        result[student_id] = test_all_houses(
            row,
            all_params
        )

    return result


def save_in_csv(
    result: dict,
    save_to_file: str
) -> None:
    """Save prediction results to a CSV file.

    Args:
        result: Dictionary containing student indices and predicted houses.
        save_to_file: Path to the output CSV file.

    Returns:
        None.
    """
    try:
        with open(
            save_to_file,
            'w',
            newline='',
            encoding='utf-8'
        ) as f:
            writer = csv.writer(f)

            # 1. header
            header = ["Index", "Hogwarts House"]
            writer.writerow(header)

            # 2. write the result
            for student_id, house_name in result.items():
                writer.writerow([
                    int(student_id),
                    house_name
                ])

        print(f"Results successfully saved to {save_to_file} !")

    except IOError as e:
        print(f"Error while writing CSV file: {e}")
        exit(1)


def logreg_predict(test_set_file: str) -> None:
    """Predict Hogwarts houses for the test dataset.

    Args:
        test_set_file: Path to the test CSV file.

    Returns:
        None.
    """
    file_model = "model_params.json"

    save_to_file = "houses.csv"

    all_params = load_model_params(file_model)

    cleaned_test_data = load_test_set(
        all_params,
        test_set_file
    )

    result = predict(
        all_params,
        cleaned_test_data
    )

    save_in_csv(
        result,
        save_to_file
    )


def main() -> None:
    """Run the logistic regression prediction program.

    Args:
        None.

    Returns:
        None.
    """
    args = sys.argv

    if len(args) != 2:
        print("Usage: ./logreg_predict.py datasets/dataset_test.csv")
        exit(1)

    test_set_file = args[1]

    try:
        logreg_predict(test_set_file)

    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
