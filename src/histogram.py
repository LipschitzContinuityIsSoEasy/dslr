#!/usr/bin/env python3

"""Displays a histogram answering the following question:

    Which Hogwarts course has a homogeneous score distribution
        between all four houses?
"""

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import validate_data, MANDATORY_COLUMNS
from matplotlib.patches import Patch


def histogram(file_name: str) -> None:
    """Display histograms for the Hogwarts dataset.

    Args:
        file_name: Path to the CSV file.

    Returns:
        None.
    """
    all_data = validate_data(file_name, testing=False)

    # trouver tous les cours, sans index ni infos eleve
    course_cols = [
        col for col in all_data.columns
        if col not in MANDATORY_COLUMNS
    ]

    # sort courses by homogeneity
    spread = most_homogeneous(all_data, course_cols)

    # show sorted courses on one grid
    show_grid(all_data, course_cols, spread)


def most_homogeneous(
    all_data: pd.DataFrame,
    course_cols: list[str]
) -> pd.Series:
    """Find the most homogeneous course.

    Args:
        all_data: Dataset containing course results and Hogwarts houses.
        course_cols: Course columns.

    Returns:
        Standard deviation of normalized house averages for each course.
    """
    course_data = all_data[course_cols]

    # standardization of data
    z_scores = (course_data - course_data.mean()) / course_data.std()

    z_scores["Hogwarts House"] = all_data["Hogwarts House"]

    # separate into groups by house
    spread = z_scores.groupby("Hogwarts House").mean().std().sort_values()

    print("Std by house :")
    print(spread.to_string())
    print(f"\nThe most homogeneous course : {spread.idxmin()}\n")

    return spread


def show_grid(
    all_data: pd.DataFrame,
    course_cols: list[str],
    spread: pd.Series
) -> None:
    """Show a grid of courses ordered by homogeneity.

    Args:
        all_data: Dataset containing course results and Hogwarts houses.
        course_cols: Course columns to display.
        spread: Standard deviation for each course.

    Returns:
        None.
    """
    # get each house
    houses = sorted(all_data["Hogwarts House"].dropna().unique())
    # create colors for each house
    palette = sns.color_palette(n_colors=len(houses))

    # get presorted courses
    ordered_courses = list(spread.index)

    # define columns and rows
    n_courses = len(ordered_courses)
    ncols = 4
    nrows = (n_courses + ncols - 1) // ncols

    # create grid
    fig, _ = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 3.5 * nrows)
    )
    # get all subplots
    axes = fig.axes

    for rank, (ax, course) in enumerate(
        # pair graph with a course
        zip(axes, ordered_courses),
        start=1
    ):
        # creates histogram
        sns.histplot(
            data=all_data,
            x=course,
            # separate by house
            hue="Hogwarts House",
            # use same house order
            hue_order=houses,
            palette=palette,
            # layer them on top of each other
            multiple="layer",
            alpha=0.5,
            stat="density",
            common_norm=False,
            kde=True,
            ax=ax,
            legend=False
        )

        ax.set_title(
            f"#{rank}  {course}  (std={spread[course]:.3f})",
            fontsize=10
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes[n_courses:]:
        ax.set_visible(False)

    # create one legend for all subplots
    handles = [
        Patch(
            facecolor=color,
            alpha=0.5,
            label=house
        )
        for house, color in zip(houses, palette)
    ]

    fig.legend(
        handles=handles,
        loc="upper right",
        ncol=len(houses)
    )

    fig.suptitle(
        "Courses ordered from most to least homogeneous",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.25)

    plt.show()


def main() -> None:
    """Show a histogram for each course, sorted by homogeneity.

    Expects the dataset path as an argument:
    ./histogram.py <dataset.csv>

    Returns:
        None.
    """
    args = sys.argv
    if len(args) < 2:
        print("Usage: ./histogram.py <dataset.csv>")
        exit(1)

    file_name = args[1]

    try:
        histogram(file_name)
    except Exception as e:
        print("Error: " + e.args[0] if e.args else "unexpected error.")
        exit(1)


if __name__ == "__main__":
    main()
