#!/usr/bin/env python3

"""Utility files to export data validation and name shortening.
"""

import pandas as pd

MANDATORY_COLUMNS = [
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand"
]

VALID_HANDS = {"Left", "Right"}

VALID_HOUSES = {"Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"}


def validate_data(
    file_name: str,
    testing: bool = False,
    all_params: dict | None = None,
) -> pd.DataFrame:
    """Load and validate the Hogwarts dataset.

    Args:
        file_name: Path to the CSV file.
        testing: Flag to mark dataset as the testing one
                 (without house labels).

    Returns:
        Validated pandas DataFrame.

    Raises:
        ValueError: If the file cannot be read or the data is invalid.
    """
    try:
        data = pd.read_csv(file_name)
    except FileNotFoundError:
        raise ValueError(f"file '{file_name}' is not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"file '{file_name}' is empty.")
    except Exception as e:
        raise ValueError(f"cannot read '{file_name}': {e}")

    if data.empty:
        raise ValueError("dataset is empty. It must have at least 1 entry.")

    missing_columns = [
        column for column in MANDATORY_COLUMNS
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"missing columns: {missing_columns}")

    if not pd.api.types.is_numeric_dtype(data["Index"]):
        raise ValueError("Index must be numeric.")
    if data["Index"].isna().any():
        raise ValueError("Index contains missing values.")
    if data["Index"].duplicated().any():
        raise ValueError("Index contains duplicate values.")

    course_columns = [
        column for column in data.columns
        if column not in MANDATORY_COLUMNS
    ]
    if not course_columns:
        raise ValueError("dataset contains no course columns.")
    for column in course_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"course '{column}' must be numeric.")
        if data[column].isna().all():
            raise ValueError(f"course '{column}' has no values.")

    invalid_hands = set(data["Best Hand"].dropna()) - VALID_HANDS
    if invalid_hands:
        raise ValueError(f"invalid Best Hand values: {invalid_hands}")

    found_houses = set(data["Hogwarts House"].dropna())
    invalid_houses = found_houses - VALID_HOUSES
    if invalid_houses:
        raise ValueError(f"invalid Hogwarts House values: {invalid_houses}")
    if not testing and not found_houses:
        raise ValueError("dataset contains no Hogwarts House values.")

    if testing:
        data = data[["Index"] + all_params["best_features"]]

    return data


def short_name(course: str) -> str:
    """Shorten a course name too long to fit beside a row.

    Args:
        course: Course name.

    Returns:
        The name itself, or its initials when it is too long.
    """
    if len(course) <= 15:
        return course

    # Defense Against the Dark Arts -> DADA
    initials = "".join(
        word[0] for word in course.split()
        if word[0].isupper()
    )
    if len(initials) > 1:
        return initials

    return course[:15]
