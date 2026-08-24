#!/usr/bin/env python3

"""Scatter plot of the two most correlated courses.

Answers the question: what are the two features that are similar?
Computes the full correlation matrix, finds the pair with the
highest Pearson coefficient, and draws them against each other
with one colour per Hogwarts house.
"""

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import validate_data, MANDATORY_COLUMNS


def scatter_plot(file_name: str) -> None:
    """Display a scatter plot of the two most similar courses.

    Args:
        file_name: Path to the CSV file.

    Returns:
        None.
    """
    #  1. lire et stocker
    # validate all the data, verify that houses exist
    all_data = validate_data(file_name, testing=False)

    # 2. trouver tous les cours
    course_cols = [
        col for col in all_data.columns
        if col not in MANDATORY_COLUMNS
    ]

    # 3. find the two courses that look alike
    course1, course2 = most_similar(all_data, course_cols)

    # 4. draw the plot
    show_scatter(all_data, course1, course2)


def most_similar(
    all_data: pd.DataFrame,
    course_cols: list[str]
) -> tuple[str, str]:
    """Find the two courses with the strongest correlation.

    Args:
        all_data: Dataset containing course results and Hogwarts houses.
        course_cols: Course columns to compare.

    Returns:
        Names of the two most similar courses.
    """
    numeric_data = all_data[course_cols]

    # pour tout afficher
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 1000)
    # pd.set_option('display.precision', 4)
    # print(numeric_data.corr())

    corr_matrix = numeric_data.corr()

    # aplatir la matrice et sort
    stacked_corr = corr_matrix.stack().sort_values(ascending=False)

    # enlever les paires d'un cours avec lui meme, elles valent 1
    filtered_corr = stacked_corr[
        stacked_corr.index.get_level_values(0)
        != stacked_corr.index.get_level_values(1)
    ]

    # print(" --- les premiers 4 corr--- ")
    # print(filtered_corr.head(4))

    best_pair = filtered_corr.index[0]
    course1, course2 = best_pair

    print(f"Les deux cours sont: {course1} et {course2}")

    return course1, course2


def show_scatter(
    all_data: pd.DataFrame,
    course1: str,
    course2: str
) -> None:
    """Draw two courses against each other, one colour per house.

    Args:
        all_data: Dataset containing course results and Hogwarts houses.
        course1: Course on the x axis.
        course2: Course on the y axis.

    Returns:
        None.
    """
    # creer le canva
    plt.figure(figsize=(8, 6))

    # dessiner
    sns.scatterplot(
        data=all_data,
        x=course1,
        y=course2,
        hue="Hogwarts House",
        alpha=0.7
    )

    plt.title(f"Scatter Plot between {course1} and {course2}")
    plt.xlabel(course1)
    plt.ylabel(course2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.show()


def main() -> None:
    """Show a scatter plot of the two most similar courses.

    Expects the dataset path as an argument:
    ./scatter_plot.py <dataset.csv>

    Returns:
        None.
    """
    args = sys.argv
    if len(args) < 2:
        print("Usage: ./scatter_plot.py <dataset.csv>")
        exit(1)

    file_name = args[1]

    try:
        scatter_plot(file_name)
    except Exception as e:
        print("Error: " + e.args[0] if e.args else "unexpected error.")
        exit(1)


if __name__ == "__main__":
    main()
