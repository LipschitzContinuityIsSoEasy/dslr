#!/usr/bin/env python3

"""Draw a training of the logistic regression, epoch after epoch.

Nothing here trains anything. logreg_train opens the window once, calls
show_progress from inside its epoch loop, and leaves the trained model
on the screen at the end:

    view = training_plot.open_view(pipeline_data)
    ...
    training_plot.show_progress(view, house, epoch, weights, bias)
    ...
    training_plot.keep_open(view)

open_view answers None when no window can be opened, and the two others
do nothing with a view of None, so a training launched without a screen
runs at full speed instead of drawing into the void.

The houses are trained one after the other, so the panels fill up house
by house: the curves and the bars of a finished house stay on screen
while the next one is being trained.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import short_name


HOUSE_COLORS = {
    "Gryffindor": "#AE0001",
    "Slytherin": "#2A623D",
    "Ravenclaw": "#222F5B",
    "Hufflepuff": "#ECB939"
}

OTHERS_COLOR = "#9A9A9A"

# a model of zeros answers 0.5 to everything, so every loss starts there
FIRST_LOSS = float(np.log(2))

# the first epochs move the most, every single one of them is drawn
EARLY_EPOCHS = 12

# then a steady pace, so the window never looks frozen
LATER_DRAWINGS = 30

# the backends able to open a window
LIVE_BACKENDS = ("qt", "tk", "macosx", "gtk", "wx", "nbagg")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Squash a linear score into a probability.

    Args:
        z: Linear scores.

    Returns:
        Probabilities between 0 and 1.
    """
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def drawn_epochs(epochs: int) -> set[int]:
    """Choose the epochs to draw.

    One drawing costs more than one epoch of training, so drawing them
    all would more than double the time the training takes. Two things
    are wanted instead: the beginning, where the model moves almost
    everything it will ever move, and a pace regular enough for the
    window to keep changing until the last epoch.

    Args:
        epochs: Number of epochs one house is trained for.

    Returns:
        Epochs worth drawing, with the first and the last one.
    """
    # never skip an epoch of the beginning
    early = set(range(min(EARLY_EPOCHS, epochs) + 1))

    # then one drawing every few epochs, whatever their number
    step = max(1, epochs // LATER_DRAWINGS)

    return early | set(range(0, epochs + 1, step)) | {epochs}


def colors_for(houses: list[str]) -> dict[str, str]:
    """Give one colour per house.

    Args:
        houses: Names of the houses.

    Returns:
        The colour of each house, its own one when it is known.
    """
    spare = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    return {
        house: HOUSE_COLORS.get(house, spare[index % len(spare)])
        for index, house in enumerate(houses)
    }


def stretch(ax: plt.Axes, needed: float, axis: str = "x") -> None:
    """Widen an axis when the model grows past it.

    The window opens before the training, so nobody knows yet how big
    the weights and the scores will get.

    Args:
        ax: Axes to widen.
        needed: Value that has to fit inside.
        axis: "x" to widen both sides, "y" to raise the top only.

    Returns:
        None.
    """
    if axis == "x":
        if needed * 1.1 > ax.get_xlim()[1]:
            ax.set_xlim(-needed * 1.25, needed * 1.25)
        return

    if needed * 1.05 > ax.get_ylim()[1]:
        ax.set_ylim(0, needed * 1.2)


# ---------------------------------------------------------------------------
# the four panels, drawn empty once
def setup_sigmoid(ax: plt.Axes) -> dict:
    """Prepare the curve the students slide along.

    One point per student, placed at its own score. The students of the
    house being trained are pushed to the right, the others to the
    left, until the two groups sit at the two ends of the curve.

    Args:
        ax: Axes of the curve.

    Returns:
        The pieces to move on every drawing.
    """
    z = np.linspace(-60, 60, 2000)
    ax.plot(z, sigmoid(z), color="black", linewidth=1, zorder=2)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)

    others = ax.scatter(
        [], [], s=9, color=OTHERS_COLOR, alpha=0.5, zorder=3
    )
    inside = ax.scatter(
        [], [], s=9, color="black", alpha=0.7, zorder=4
    )

    ax.set_xlim(-6, 6)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("Score of the student", fontsize=9)
    ax.set_ylabel("Probability of being in the house", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)

    return {"axes": ax, "inside": inside, "others": others}


def setup_losses(ax: plt.Axes, houses: list[str], epochs: int) -> dict:
    """Prepare one falling log loss curve per house.

    Args:
        ax: Axes of the losses.
        houses: Names of the houses.
        epochs: Number of epochs one house is trained for.

    Returns:
        The pieces to move on every drawing.
    """
    colors = colors_for(houses)

    lines = {
        house: ax.plot(
            [], [], color=colors[house], linewidth=1.8, label=house
        )[0]
        for house in houses
    }

    ax.set_xlim(0, epochs)
    ax.set_ylim(0, FIRST_LOSS * 1.05)
    ax.set_xlabel("Epoch of this house", fontsize=9)
    ax.set_ylabel("Log loss", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("How wrong each house's model still is", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")

    return {"axes": ax, "lines": lines}


def setup_weights(
    ax: plt.Axes,
    houses: list[str],
    features: list[str]
) -> dict:
    """Prepare the bars of the weights of every course.

    Args:
        ax: Axes of the weights.
        houses: Names of the houses.
        features: Courses the model is trained on.

    Returns:
        The pieces to move on every drawing.
    """
    colors = colors_for(houses)

    positions = np.arange(len(features))
    height = 0.8 / len(houses)

    bars = {}
    for index, house in enumerate(houses):
        offset = (index - (len(houses) - 1) / 2) * height
        bars[house] = ax.barh(
            positions + offset,
            np.zeros(len(features)),
            height=height,
            color=colors[house],
            label=house
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [short_name(feature) for feature in features],
        fontsize=7
    )
    ax.invert_yaxis()

    # everything starts at zero, stretch widens this later
    ax.set_xlim(-0.5, 0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Weight", fontsize=9)
    ax.set_title("What each house learns to look at", fontsize=10)

    return {"axes": ax, "bars": bars}


def setup_accuracy(ax: plt.Axes, houses: list[str], epochs: int) -> dict:
    """Prepare the climbing accuracy curve.

    One house alone cannot sort anybody: its model only says yes or no.
    The curve therefore climbs by steps, one step every time a house
    joins the ones already trained.

    Args:
        ax: Axes of the accuracy.
        houses: Names of the houses.
        epochs: Number of epochs one house is trained for.

    Returns:
        The pieces to move on every drawing.
    """
    line = ax.plot([], [], color="#1f6f6f", linewidth=2)[0]

    label = ax.text(
        0.5, 0.3, "",
        transform=ax.transAxes, fontsize=26, ha="center",
        color="#1f6f6f", weight="bold"
    )

    # one mark per house, where its training starts
    for index in range(1, len(houses)):
        ax.axvline(
            index * epochs, color="grey", linewidth=0.8, alpha=0.5
        )

    ax.set_xlim(0, len(houses) * epochs)
    ax.set_ylim(0, 1.02)
    ax.axhline(0.98, color="grey", linestyle="--", linewidth=0.8)
    ax.text(
        0.99, 0.975, "98% asked by the subject",
        transform=ax.get_yaxis_transform(), fontsize=7,
        color="grey", ha="right", va="top"
    )
    ax.set_xlabel("Epochs trained, all houses together", fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("Students sorted in the right house", fontsize=10)

    return {"axes": ax, "line": line, "label": label}


# ---------------------------------------------------------------------------
# what logreg_train calls
def open_view(pipeline_data: dict) -> dict | None:
    """Open the window before the training starts.

    Args:
        pipeline_data: The normalized dataset logreg_train is about to
            train on.

    Returns:
        The view to give back to show_progress and keep_open, or None
        when no window can be opened.
    """
    if not plt.get_backend().lower().startswith(LIVE_BACKENDS):
        # nothing would be shown, so do not slow the training down
        print(
            f"Warning: backend '{plt.get_backend()}' opens no window, "
            "training without the animation."
        )
        return None

    normalized_data = pipeline_data["normalized_data"]
    features = pipeline_data["best_features"]
    epochs = pipeline_data["epochs"]

    houses = sorted(set(normalized_data["Hogwarts House"]))

    # a window has to be alive before anything is drawn in it
    plt.ion()

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))

    view = {
        "figure": figure,
        "option": pipeline_data["option"],
        "epochs": epochs,
        "drawn": drawn_epochs(epochs),
        "houses": houses,
        "colors": colors_for(houses),
        "students": normalized_data[features].to_numpy(dtype=float),
        "true_houses": normalized_data["Hogwarts House"].to_numpy(),
        "sigmoid": setup_sigmoid(axes[0][0]),
        "losses": setup_losses(axes[0][1], houses, epochs),
        "weights": setup_weights(axes[1][0], houses, features),
        "accuracy": setup_accuracy(axes[1][1], houses, epochs),
        # what has been seen so far, to keep the curves growing
        "scores": {},
        "seen_losses": {house: ([], []) for house in houses},
        "seen_accuracy": ([], []),
        "house": None,
        "done_epochs": 0
    }

    if figure.canvas.manager is not None:
        figure.canvas.manager.set_window_title("Training")

    figure.tight_layout(rect=(0, 0, 1, 0.95))

    plt.show(block=False)

    return view


def show_progress(
    view: dict | None,
    house: str,
    epoch: int,
    weights: list[float],
    bias: float
) -> None:
    """Draw one epoch, from inside the training loop.

    Called for every epoch of every house, and draws only the ones
    worth drawing. Does nothing at all without a view, so the training
    is free to call it whether the animation was asked for or not.

    Args:
        view: Output of open_view, or None to draw nothing.
        house: House whose model is being trained.
        epoch: Number of epochs this house has done.
        weights: One weight per course, as the training holds them.
        bias: Bias of this house.

    Returns:
        None.
    """
    if view is None or epoch not in view["drawn"]:
        return

    if house != view["house"]:
        start_house(view, house)

    scores = view["students"] @ np.asarray(weights, dtype=float) + bias
    view["scores"][house] = scores

    draw_sigmoid(view, house, scores)
    draw_losses(view, house, epoch, scores)
    draw_weights(view, house, weights)
    draw_accuracy(view, epoch)

    view["figure"].suptitle(
        f"{view['option']}   -   {house}   -   "
        f"epoch {epoch} / {view['epochs']}",
        fontsize=15,
        weight="bold"
    )

    draw_now(view)


def start_house(view: dict, house: str) -> None:
    """Move on to the house that just started being trained.

    Args:
        view: Output of open_view.
        house: House whose model is now being trained.

    Returns:
        None.
    """
    if view["house"] is not None:
        view["done_epochs"] += view["epochs"]

    view["house"] = house

    ax = view["sigmoid"]["axes"]
    view["sigmoid"]["inside"].set_color(view["colors"][house])

    ax.set_title(f"{house} against all the others", fontsize=10)
    ax.legend(
        [view["sigmoid"]["inside"], view["sigmoid"]["others"]],
        [house, "other houses"],
        fontsize=7,
        loc="upper left"
    )


def draw_sigmoid(view: dict, house: str, scores: np.ndarray) -> None:
    """Place every student on the curve, at its own score.

    Args:
        view: Output of open_view.
        house: House being trained.
        scores: Score of every student for this house.

    Returns:
        None.
    """
    inside = view["true_houses"] == house
    probabilities = sigmoid(scores)

    view["sigmoid"]["inside"].set_offsets(
        np.column_stack([scores[inside], probabilities[inside]])
    )
    view["sigmoid"]["others"].set_offsets(
        np.column_stack([scores[~inside], probabilities[~inside]])
    )

    stretch(view["sigmoid"]["axes"], float(np.abs(scores).max()))


def draw_losses(
    view: dict,
    house: str,
    epoch: int,
    scores: np.ndarray
) -> None:
    """Add one point to the curve of the house being trained.

    Args:
        view: Output of open_view.
        house: House being trained.
        epoch: Number of epochs this house has done.
        scores: Score of every student for this house.

    Returns:
        None.
    """
    # log(0) is not a number, so stay just inside the borders
    safe = np.clip(sigmoid(scores), 1e-15, 1.0 - 1e-15)
    y = (view["true_houses"] == house).astype(float)

    loss = float(
        -(y * np.log(safe) + (1.0 - y) * np.log(1.0 - safe)).mean()
    )

    epochs, losses = view["seen_losses"][house]
    epochs.append(epoch)
    losses.append(loss)

    view["losses"]["lines"][house].set_data(epochs, losses)

    stretch(view["losses"]["axes"], loss, "y")


def draw_weights(view: dict, house: str, weights: list[float]) -> None:
    """Move the bars of the house being trained.

    The bars of the houses already trained keep their last width.

    Args:
        view: Output of open_view.
        house: House being trained.
        weights: One weight per course.

    Returns:
        None.
    """
    for bar, weight in zip(view["weights"]["bars"][house], weights):
        bar.set_width(weight)

    stretch(view["weights"]["axes"], float(np.abs(weights).max()))


def draw_accuracy(view: dict, epoch: int) -> None:
    """Sort every student with the houses trained so far.

    Args:
        view: Output of open_view.
        epoch: Number of epochs the current house has done.

    Returns:
        None.
    """
    known = list(view["scores"])

    # the house with the highest probability wins, like test_all_houses
    together = np.column_stack([view["scores"][house] for house in known])
    chosen = np.array(known)[together.argmax(axis=1)]

    accuracy = float((chosen == view["true_houses"]).mean())

    epochs, accuracies = view["seen_accuracy"]
    epochs.append(view["done_epochs"] + epoch)
    accuracies.append(accuracy)

    view["accuracy"]["line"].set_data(epochs, accuracies)
    view["accuracy"]["label"].set_text(f"{accuracy:.1%}")


def draw_now(view: dict) -> None:
    """Push what was just changed to the window.

    Args:
        view: Output of open_view.

    Returns:
        None.
    """
    view["figure"].canvas.draw()
    view["figure"].canvas.flush_events()

    # lets the window answer the mouse between two epochs
    plt.pause(0.001)


def keep_open(view: dict | None) -> None:
    """Leave the trained model on the screen.

    Args:
        view: Output of open_view, or None when nothing was drawn.

    Returns:
        None.
    """
    if view is None:
        return

    print("Close the window to quit.")

    plt.ioff()
    plt.show(block=True)
