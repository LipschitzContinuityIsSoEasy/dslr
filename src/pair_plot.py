#!/usr/bin/env python3

"""Pair plot of every course, coloured by Hogwarts house.

Answers the question: from this visualisation, which features
are you going to use for your logistic regression?  Draws a
zoomable pair plot with KDE diagonals and ranks the courses by
ANOVA F-score to highlight the most discriminating ones.
"""

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import validate_data, MANDATORY_COLUMNS, short_name
from sklearn.feature_selection import f_classif


def pair_plot(file_name: str) -> None:
    """Display a pair plot of the Hogwarts dataset.

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

    # 3. draw the plot
    show_pair_plot(all_data, course_cols)

    # 4. rank courses by F score
    rank_courses(all_data, course_cols)


def show_pair_plot(
    all_data: pd.DataFrame,
    course_cols: list[str]
) -> None:
    """Draw the pair plot of every course in a zoomable window.

    Args:
        all_data: Dataset containing course results and Hogwarts houses.
        course_cols: Course columns to display.

    Returns:
        None.
    """
    # ajouter House dans course, faciliter seaborn
    features_to_plot = course_cols + ['Hogwarts House']
    plot_data = all_data[features_to_plot].dropna()

    # turns on short names
    plot_data = plot_data.rename(columns={
        col: short_name(col) for col in course_cols
    })

    print(
        "Génération du pair plot en cours "
        "(cela peut prendre quelques secondes)..."
    )

    grid = sns.pairplot(
        plot_data,
        hue="Hogwarts House",
        diag_kind="kde",
        palette="Set2",
        height=0.9
    )

    show_zoomable(grid.figure)

    plt.show()


def show_zoomable(figure: plt.Figure) -> None:
    """Let a figure be dragged and zoomed.

    Every case keeps its own axes, so the grid stays sharp at any zoom.
    Cases pushed outside the window are hidden and not drawn for speed.

    Args:
        figure: Figure to make navigable.

    Returns:
        None.
    """
    # position of every case, in fractions of the figure
    base = [(ax, ax.get_position().frozen()) for ax in figure.axes]

    # zoom level and offset of the whole grid
    view = {"zoom": 1.0, "x": 0.0, "y": 0.0}
    grab = {}

    small_label = 7
    small_tick = 5

    if figure.canvas.manager is not None:
        figure.canvas.manager.set_window_title(
            "Pair plot"
        )

    def apply_view() -> None:
        zoom, x, y = view["zoom"], view["x"], view["y"]

        label_size = min(small_label * zoom, 11)
        tick_size = min(small_tick * zoom, 9)

        for ax, position in base:
            left = position.x0 * zoom + x
            bottom = position.y0 * zoom + y
            width = position.width * zoom
            height = position.height * zoom

            # hidden cases are not drawn
            is_visible = (
                left < 1 and left + width > 0
                and bottom < 1 and bottom + height > 0
            )
            ax.set_visible(is_visible)

            if is_visible:
                ax.set_position([left, bottom, width, height])
                ax.tick_params(labelsize=tick_size)
                ax.xaxis.label.set_size(label_size)
                ax.yaxis.label.set_size(label_size)

        figure.canvas.draw_idle()

    def on_scroll(event) -> None:
        # the wheel zooms around the cursor
        factor = 1.25 if event.button == "up" else 0.8
        zoom = view["zoom"] * factor

        if zoom <= 1.0:
            # never smaller than the whole grid
            view.update({"zoom": 1.0, "x": 0.0, "y": 0.0})
        else:
            # keep the point under the cursor
            x = event.x / figure.bbox.width
            y = event.y / figure.bbox.height

            view["x"] = x - (x - view["x"]) * factor
            view["y"] = y - (y - view["y"]) * factor
            view["zoom"] = zoom

        apply_view()

    def on_press(event) -> None:
        # the toolbar has priority when one of its modes is active
        toolbar = figure.canvas.toolbar
        if toolbar is not None and toolbar.mode:
            return

        # nothing to move while the whole grid is on screen
        if event.button == 1 and view["zoom"] > 1.0:
            grab["x"] = event.x
            grab["y"] = event.y

    def on_motion(event) -> None:
        # move the grid by the same distance as the cursor
        if not grab or event.x is None:
            return

        view["x"] += (event.x - grab["x"]) / figure.bbox.width
        view["y"] += (event.y - grab["y"]) / figure.bbox.height

        grab["x"] = event.x
        grab["y"] = event.y

        apply_view()

    def on_release(event) -> None:
        grab.clear()

    # shrink the labels for the first view
    apply_view()

    figure.canvas.mpl_connect("scroll_event", on_scroll)
    figure.canvas.mpl_connect("button_press_event", on_press)
    figure.canvas.mpl_connect("motion_notify_event", on_motion)
    figure.canvas.mpl_connect("button_release_event", on_release)


def rank_courses(
    all_data: pd.DataFrame,
    course_cols: list[str]
) -> None:
    """Rank the courses by their ability to separate the houses.

    Args:
        all_data: Dataset containing course results and Hogwarts houses.
        course_cols: Course columns to rank.

    Returns:
        None.
    """
    # pour f-score
    clean_data = all_data[course_cols + ['Hogwarts House']].dropna()

    X = clean_data[course_cols]
    y = clean_data['Hogwarts House']

    # 4. Calculer la valeur F (F-score) et la valeur P pour chaque cours
    f_values, p_values = f_classif(X, y)

    # 5. Regrouper dans un DataFrame et trier par valeur F décroissante
    feature_ranking = pd.DataFrame({
        'Feature': course_cols,
        'F_Value': f_values,
        'P_Value': p_values
    }).sort_values(by='F_Value', ascending=False).reset_index(drop=True)

    print(
        "--- Classement des cours selon leur capacité à différencier les "
        "maisons (plus le F-score est élevé, meilleur est le résultat de "
        "classification) ---"
    )
    print(feature_ranking.to_string(index=False))

    # 6. Sélectionner automatiquement les 10 meilleures caractéristiques
    # avec le F-score le plus élevé pour les utiliser dans la régression
    # logistique
    top_k = 10
    best_features = feature_ranking['Feature'].head(top_k).tolist()
    print(
        f"\nCaractéristiques recommandées pour la régression logistique "
        f"(top {top_k}) :"
    )
    print(best_features)


def main() -> None:
    """Show a pair plot of the dataset and rank the courses.

    Expects the dataset path as an argument:
    ./pair_plot.py <dataset.csv>

    Returns:
        None.
    """
    args = sys.argv
    if len(args) < 2:
        print("Usage: ./pair_plot.py <dataset.csv>")
        exit(1)

    file_name = args[1]

    try:
        pair_plot(file_name)
    except Exception as e:
        print("Error: " + e.args[0] if e.args else "unexpected error.")
        exit(1)


if __name__ == "__main__":
    main()
