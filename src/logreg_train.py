#!/usr/bin/env python3

import sys
import pandas as pd
import math
import json
import training_plot

from sklearn.feature_selection import f_classif
from utils import validate_data, MANDATORY_COLUMNS


def calculate_mean(values: pd.Series) -> float:
    """Calculate the mean of a series.

    Args:
        values: Numeric values used to calculate the mean.

    Returns:
        The mean of the values.
    """
    return values.sum() / values.count()


def fill_missing_values(
        data: pd.DataFrame,
        course_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fill missing course values with their column means.

    Args:
        data: Dataset containing course columns.
        course_cols: List of course column names.

    Returns:
        A tuple containing the cleaned data and the original course means.
    """
    raw_means = {}

    for col in course_cols:
        mean_val = calculate_mean(data[col])

        raw_means[col] = mean_val

        data[col] = data[col].fillna(mean_val)

    return data, raw_means


def select_best_features(
    data: pd.DataFrame,
    course_cols: list[str]
) -> list[str]:
    """Select the ten course features with the highest F-scores.

    Args:
        data: Dataset containing the course values and house labels.
        course_cols: List of course column names.

    Returns:
        List of the ten best course features.
    """
    clean_data = data[
        course_cols + ["Hogwarts House"]
    ].dropna(subset=["Hogwarts House"])

    X = clean_data[course_cols]

    y = clean_data["Hogwarts House"]

    f_values, p_values = f_classif(X, y)

    feature_ranking = pd.DataFrame({
        "Feature": course_cols,
        "F_Value": f_values,
        "P_Value": p_values
    }).sort_values(
        by="F_Value",
        ascending=False
    ).reset_index(drop=True)

    print(
        "--- Course ranking by ability to differentiate houses ---"
    )
    print(feature_ranking.to_string(index=False))

    top_k = 10

    best_features = (
        feature_ranking["Feature"]
        .head(top_k)
        .tolist()
    )

    print(
        f"\nRecommended features for logistic regression "
        f"(top {top_k}):"
    )
    print(best_features)

    return best_features


def load_and_preprocess_data(
    file_name: str,
    pipeline_data: dict
) -> dict:
    """Load, clean, and prepare data for logistic regression.

    Args:
        file_name: Path to the training CSV file.
        pipeline_data: Dictionary containing training parameters.

    Returns:
        Updated pipeline data containing the cleaned dataset,
        selected features, means, and standard deviations.
    """
    all_data = validate_data(file_name, False)

    course_cols = [
        col
        for col in all_data.columns
        if col not in MANDATORY_COLUMNS
    ]

    all_data, raw_means = fill_missing_values(
        all_data,
        course_cols
    )

    best_features = select_best_features(
        all_data,
        course_cols
    )

    clean_data = all_data[
        course_cols + ["Hogwarts House"]
    ].dropna(subset=["Hogwarts House"])

    train_means = {
        col: raw_means[col]
        for col in best_features
    }

    train_stds = {}

    for col in best_features:
        std_val = clean_data[col].std()

        if std_val == 0:
            std_val = 1.0

        train_stds[col] = std_val

    pipeline_data.update({
        "cleaned_data": clean_data[
            best_features + ["Hogwarts House"]
        ],
        "best_features": best_features,
        "train_means": train_means,
        "train_stds": train_stds
    })

    return pipeline_data


def normalize_data(pipeline_data: dict) -> dict:
    """Normalize the selected features using Z-score standardization.

    Args:
        pipeline_data: Dictionary containing the cleaned data and parameters.

    Returns:
        The updated pipeline data containing the normalized dataset.
    """
    cleaned_data = pipeline_data["cleaned_data"]

    best_features = pipeline_data["best_features"]

    train_means = pipeline_data["train_means"]

    train_stds = pipeline_data["train_stds"]

    # 1. Copy the cleaned data
    normalized_copie = cleaned_data.copy()

    # 2. Remove the house labels before normalization
    houses = normalized_copie.pop('Hogwarts House')

    # 3. Normalize the selected features
    for col in best_features:
        mean_val = train_means[col]

        std_val = train_stds[col]

        normalized_copie[col] = (
            normalized_copie[col] - mean_val
        ) / std_val

    # 4. Add the house labels back
    normalized_copie['Hogwarts House'] = houses

    # 5. Replace cleaned_data with normalized_data
    pipeline_data.pop("cleaned_data", None)

    pipeline_data["normalized_data"] = normalized_copie

    return pipeline_data


def calculate_single_prediction_and_error(
    row: pd.Series,
    best_features: list[str],
    weights: list[float],
    bias: float,
    target_house: str
) -> float:
    """Calculate the prediction error for one training sample.

    Args:
        row: DataFrame row containing the sample features and house label.
        best_features: List of feature names used for training.
        weights: Model weights for the selected features.
        bias: Model bias.
        target_house: Hogwarts house used as the positive class.

    Returns:
        The prediction error for the sample.
    """
    num_features = len(best_features)

    # 1. Calculate the linear score: z = b + w1*x1 + w2*x2 + ...
    z = bias

    for j in range(num_features):
        feature_name = best_features[j]

        z += weights[j] * row[feature_name]

    # 2. Protect against numerical overflow in the sigmoid calculation
    if z < -700:
        prediction = 0.0

    elif z > 700:
        prediction = 1.0

    else:
        prediction = 1.0 / (1.0 + math.exp(-z))

    # 3. Get the real label
    real_house = row['Hogwarts House']

    y = 0.0

    if real_house == target_house:
        y = 1.0

    # 4. Calculate the prediction error
    error = prediction - y

    return error


def get_optimizer_settings(
    option: str,
    data_size: int
) -> tuple[int, bool]:
    """Get batch size and shuffle settings for the optimizer.

    Args:
        option: Gradient descent optimizer to use.
        data_size: Number of training samples.

    Returns:
        A tuple containing the batch size and shuffle setting.

    Raises:
        ValueError: If the specified optimizer is unknown.
    """
    if option == "BGD":
        return data_size, False

    elif option == "SGD":
        return 1, True

    elif option == "minibatch":
        return 32, True

    elif option == "minibatch-noshuffle":
        return 32, False

    else:
        raise ValueError(f"Unknown optimizer option: {option}")


def train_single_house(
    pipeline_data: dict,
    target_house: str,
    view=None
) -> tuple[list[float], float]:
    """Train a binary logistic regression model for one house.

    Args:
        pipeline_data: Dictionary containing training data and parameters.
        target_house: Name of the Hogwarts house to train the model for.

    Returns:
        A tuple containing the trained weights and bias.

    Raises:
        ValueError: If the specified optimizer is unknown.
    """
    option = pipeline_data["option"]

    normalized_data = pipeline_data["normalized_data"]

    best_features = pipeline_data["best_features"]

    learning_rate = pipeline_data["learning_rate"]

    epochs = pipeline_data["epochs"]

    m = len(normalized_data)

    num_features = len(best_features)

    weights = [0.0] * num_features

    bias = 0.0

    batch_size, shuffle = get_optimizer_settings(
        option,
        m
    )

    training_plot.show_progress(view, target_house, 0, weights, bias)

    for epoch in range(epochs):
        if shuffle:
            current_data = (
                normalized_data
                .sample(frac=1)
                .reset_index(drop=True)
            )

        else:
            current_data = normalized_data

        # # stocker l'érreur carrée totale pour cet itération
        # current_sum_mse = 0.0
        for i in range(0, m, batch_size):
            batch = current_data.iloc[i: i + batch_size]

            current_batch_size = len(batch)

            sum_error_weights = [0.0] * num_features

            sum_error_bias = 0.0

            for _, row in batch.iterrows():
                error = calculate_single_prediction_and_error(
                    row,
                    best_features,
                    weights,
                    bias,
                    target_house
                )

                sum_error_bias += error

                for j in range(num_features):
                    feature_name = best_features[j]

                    sum_error_weights[j] += (
                        error * row[feature_name]
                    )

            bias = (
                bias
                - (
                    learning_rate
                    * (1 / current_batch_size)
                    * sum_error_bias
                )
            )

            for j in range(num_features):
                weights[j] = (
                    weights[j]
                    - (
                        learning_rate
                        * (1 / current_batch_size)
                        * sum_error_weights[j]
                    )
                )

        training_plot.show_progress(
            view, target_house, epoch, weights, bias
        )

    return weights, bias


def save_all_to_json(all_model_data: dict, filename_json: str) -> None:
    """Save all trained model parameters.

    This includes features, means, stds, weights, and biases
    into a JSON file with proper error handling.

    Args:
        all_model_data: Dictionary containing the trained model parameters.
        filename_json: Path to the JSON file.

    Returns:
        None.

    Raises:
        FileNotFoundError: If the file or directory is not found.
        PermissionError: If permission is denied when writing the file.
        IOError: If an input/output error occurs.
        Exception: If an unexpected error occurs.
    """
    try:
        # Serialize and write model data to JSON
        with open(filename_json, "w") as f:
            json.dump(all_model_data, f, indent=4)

        print(
            f"Complete model successfully saved to '{filename_json}'!"
        )

    except FileNotFoundError:
        print(
            f"Error: The file '{filename_json}' was not found."
        )
        exit(1)

    except PermissionError:
        print(
            f"Error: Permission denied for file '{filename_json}'."
        )
        exit(1)

    except IOError as e:
        print(
            f"Error: I/O error while saving the JSON file: {e}"
        )
        exit(1)

    except Exception as e:
        print(
            f"Error: An unexpected error occurred: {e}"
        )
        exit(1)


def train_all_houses(pipeline_data: dict, view=None) -> None:
    """Train One-vs-All (OvR) binary classifiers for all 4 Hogwarts houses
    and save the combined model parameters to a JSON file.

    Args:
        pipeline_data: Dictionary containing the training data and parameters.

    Returns:
        None.

    Raises:
        Exception: If training or saving the model fails.
    """
    houses = [
        'Gryffindor',
        'Slytherin',
        'Ravenclaw',
        'Hufflepuff'
    ]

    filename_json = "model_params.json"

    best_features = pipeline_data["best_features"]

    train_means = pipeline_data["train_means"]

    train_stds = pipeline_data["train_stds"]

    weights_bias = {}

    for house in houses:
        print(f"Training model for: {house}...")

        weights, bias = train_single_house(
            pipeline_data,
            house,
            view
        )

        weights_dict = {}

        for j in range(len(best_features)):
            feature_name = best_features[j]

            weight_value = weights[j]

            weights_dict[feature_name] = weight_value

        weights_bias[house] = {
            "weights": weights_dict,
            "bias": bias
        }

    all_model_data = {
        "best_features": best_features,
        "train_means": train_means,
        "train_stds": train_stds,
        "weights": weights_bias
    }

    # Save all trained parameters to JSON
    save_all_to_json(all_model_data, filename_json)

    print("All models trained successfully!")


def logreg_train(file_name: str, option: str) -> None:
    """Train a logistic regression model.

    Args:
        file_name: Path to the training dataset.
        option: Gradient descent optimizer to use.

    Returns:
        None.

    Raises:
        Exception: If training fails.
    """
    pipeline_data = {
        "option": option,
        "learning_rate": 0.1,
        "epochs": 300,
        "batch_size": 32
    }

    # 1. Load and preprocess data
    pipeline_data = load_and_preprocess_data(
        file_name,
        pipeline_data
    )

    # 2. Normalize features (Z-score)
    pipeline_data = normalize_data(pipeline_data)

    # 3. Open animating window
    view = training_plot.open_view(pipeline_data)

    # 4. Train all houses and save model parameters
    train_all_houses(pipeline_data, view)

    # 5. Keep the animation window open
    training_plot.keep_open(view)


def main() -> None:
    """Train a logistic regression model using the specified optimizer.

    Expects the dataset path and an optional optimizer as arguments:

    ./logreg_train.py <dataset_path.csv> \
        [BGD/SGD/minibatch/minibatch-noshuffle]

    Args:
        None.

    Returns:
        None.

    Raises:
        Exception: If training fails.
    """
    args = sys.argv

    if len(args) < 2:
        print(
            "Usage: ./logreg_train.py datasets/dataset_train.csv [option]"
        )
        print(
            "Optimizers available: BGD (default), SGD, minibatch, "
            "minibatch-noshuffle"
        )
        exit(1)

    file_name = args[1]

    if len(args) > 2:
        option = args[2]

    else:
        option = "BGD"

    try:
        logreg_train(file_name, option)

    except Exception as e:
        print(
            "Error: " + e.args[0]
            if e.args
            else "unexpected error."
        )
        exit(1)


if __name__ == "__main__":
    main()
