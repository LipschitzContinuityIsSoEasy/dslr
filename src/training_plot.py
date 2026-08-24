#!/usr/bin/env python3

"""Live training animation for the Sorting Hat classifier.

Shows four panels during training: the sigmoid curve with students,
the log-loss per house, the weight bars, and the overall accuracy.
logreg_train drives the animation:

    view = training_plot.open_view(pipeline_data)
    ...
    training_plot.show_progress(view, house, epoch, weights, bias)
    ...
    training_plot.keep_open(view)

When no display is available, open_view returns None and the other
two functions do nothing, so training runs at full speed.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils import short_name


HOUSE_COLORS = {
    "Gryffindor": "#AE0001",
    "Slytherin": "#2A623D",
    "Ravenclaw": "#222F5B",
    "Hufflepuff": "#ECB939",
}

LIVE_BACKENDS = ("qt", "tk", "macosx", "gtk", "wx", "nbagg")


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Squash linear scores into probabilities."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def _should_draw(epoch: int, epochs: int) -> bool:
    """Decide whether this epoch is worth drawing."""
    if epoch == 0 or epoch == epochs - 1:
        return True
    if epoch <= 8:
        return epoch % 2 == 0
    if epoch <= 20:
        return epoch % 8 == 0
    if epoch <= 50:
        return epoch % 10 == 0
    if epoch <= 80:
        return epoch % 40 == 0
    return epoch % 90 == 0


def _house_color(house: str, index: int) -> str:
    """Return the colour for a house."""
    if house in HOUSE_COLORS:
        return HOUSE_COLORS[house]
    fallback = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return fallback[index % len(fallback)]


def open_view(pipeline_data: dict) -> dict | None:
    """Open the training window.

    Args:
        pipeline_data: Normalized dataset and training parameters.

    Returns:
        A view dict for show_progress and keep_open, or None
        when no display is available.
    """
    if not plt.get_backend().lower().startswith(LIVE_BACKENDS):
        print(
            f"Warning: backend '{plt.get_backend()}' opens no window, "
            "training without the animation."
        )
        return None

    data = pipeline_data["normalized_data"]
    features = pipeline_data["best_features"]
    epochs = pipeline_data["epochs"]
    houses = sorted(set(data["Hogwarts House"]))
    colors = {
        h: _house_color(h, i) for i, h in enumerate(houses)
    }

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))

    # --- sigmoid panel ---
    ax_sig = axes[0][0]
    z = np.linspace(-60, 60, 2000)
    ax_sig.plot(z, sigmoid(z), color="black", linewidth=1)
    ax_sig.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    inside_dots = ax_sig.scatter([], [], s=9, alpha=0.7, zorder=4)
    others_dots = ax_sig.scatter(
        [], [], s=9, color="#9A9A9A", alpha=0.5, zorder=3
    )
    ax_sig.set_xlim(-6, 6)
    ax_sig.set_ylim(-0.04, 1.04)
    ax_sig.set_xlabel("Score")
    ax_sig.set_ylabel("Probability")
    ax_sig.grid(alpha=0.25)

    # --- loss panel ---
    ax_loss = axes[0][1]
    loss_lines = {
        h: ax_loss.plot([], [], color=colors[h], linewidth=1.8, label=h)[0]
        for h in houses
    }
    ax_loss.set_xlim(0, epochs)
    ax_loss.set_ylim(0, np.log(2) * 1.05)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Log loss")
    ax_loss.set_title("Loss per house")
    ax_loss.legend(fontsize=7, loc="upper right")
    ax_loss.grid(alpha=0.25)

    # --- weights panel ---
    ax_w = axes[1][0]
    positions = np.arange(len(features))
    bar_h = 0.8 / len(houses)
    weight_bars = {}
    for i, h in enumerate(houses):
        offset = (i - (len(houses) - 1) / 2) * bar_h
        weight_bars[h] = ax_w.barh(
            positions + offset, np.zeros(len(features)),
            height=bar_h, color=colors[h], label=h,
        )
    ax_w.set_yticks(positions)
    ax_w.set_yticklabels([short_name(f) for f in features], fontsize=7)
    ax_w.invert_yaxis()
    ax_w.set_xlim(-0.5, 0.5)
    ax_w.axvline(0, color="black", linewidth=0.8)
    ax_w.set_xlabel("Weight")
    ax_w.set_title("Learned weights")

    # --- accuracy panel ---
    ax_acc = axes[1][1]
    acc_line = ax_acc.plot([], [], color="#1f6f6f", linewidth=2)[0]
    acc_label = ax_acc.text(
        0.5, 0.3, "", transform=ax_acc.transAxes,
        fontsize=26, ha="center", color="#1f6f6f", weight="bold",
    )
    for i in range(1, len(houses)):
        ax_acc.axvline(i * epochs, color="grey", linewidth=0.8, alpha=0.5)
    ax_acc.axhline(0.98, color="grey", linestyle="--", linewidth=0.8)
    ax_acc.set_xlim(0, len(houses) * epochs)
    ax_acc.set_ylim(0, 1.02)
    ax_acc.set_xlabel("Total epochs")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Training accuracy")
    ax_acc.grid(alpha=0.25)

    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title("Training")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show(block=False)

    return {
        "fig": fig,
        "option": pipeline_data["option"],
        "epochs": epochs,
        "houses": houses,
        "colors": colors,
        "students": data[features].to_numpy(dtype=float),
        "true_houses": data["Hogwarts House"].to_numpy(),
        # artists
        "inside_dots": inside_dots,
        "others_dots": others_dots,
        "ax_sig": ax_sig,
        "loss_lines": loss_lines,
        "ax_loss": ax_loss,
        "weight_bars": weight_bars,
        "ax_w": ax_w,
        "acc_line": acc_line,
        "acc_label": acc_label,
        # accumulated state
        "scores": {},
        "loss_data": {h: ([], []) for h in houses},
        "acc_data": ([], []),
        "current_house": None,
        "done_epochs": 0,
    }


def show_progress(
    view: dict | None,
    house: str,
    epoch: int,
    weights: list[float],
    bias: float,
) -> None:
    """Draw one epoch from inside the training loop.

    Args:
        view: Output of open_view, or None to skip drawing.
        house: House being trained.
        epoch: Current epoch for this house.
        weights: Model weights.
        bias: Model bias.
    """
    if view is None or not _should_draw(epoch, view["epochs"]):
        return

    # detect house switch
    if house != view["current_house"]:
        if view["current_house"] is not None:
            view["done_epochs"] += view["epochs"]
        view["current_house"] = house
        view["inside_dots"].set_color(view["colors"][house])
        view["ax_sig"].set_title(f"{house} vs rest")

    scores = view["students"] @ np.asarray(weights) + bias
    view["scores"][house] = scores
    probs = sigmoid(scores)
    inside = view["true_houses"] == house

    # sigmoid panel
    view["inside_dots"].set_offsets(
        np.column_stack([scores[inside], probs[inside]])
    )
    view["others_dots"].set_offsets(
        np.column_stack([scores[~inside], probs[~inside]])
    )
    peak = float(np.abs(scores).max()) * 1.25
    if peak > view["ax_sig"].get_xlim()[1]:
        view["ax_sig"].set_xlim(-peak, peak)

    # loss panel
    safe = np.clip(probs, 1e-15, 1 - 1e-15)
    y = inside.astype(float)
    loss = float(-(y * np.log(safe) + (1 - y) * np.log(1 - safe)).mean())
    ex, ey = view["loss_data"][house]
    ex.append(epoch)
    ey.append(loss)
    view["loss_lines"][house].set_data(ex, ey)
    if loss > view["ax_loss"].get_ylim()[1] * 0.95:
        view["ax_loss"].set_ylim(0, loss * 1.2)

    # weights panel
    for bar, w in zip(view["weight_bars"][house], weights):
        bar.set_width(w)
    peak_w = float(np.abs(weights).max()) * 1.25
    if peak_w > view["ax_w"].get_xlim()[1]:
        view["ax_w"].set_xlim(-peak_w, peak_w)

    # accuracy panel
    known = list(view["scores"])
    together = np.column_stack([view["scores"][h] for h in known])
    chosen = np.array(known)[together.argmax(axis=1)]
    acc = float((chosen == view["true_houses"]).mean())
    ax, ay = view["acc_data"]
    ax.append(view["done_epochs"] + epoch)
    ay.append(acc)
    view["acc_line"].set_data(ax, ay)
    view["acc_label"].set_text(f"{acc:.1%}")

    # title and flush
    view["fig"].suptitle(
        f"{view['option']}  —  {house}  —  "
        f"epoch {epoch}/{view['epochs']}",
        fontsize=15, weight="bold",
    )
    view["fig"].canvas.draw()
    view["fig"].canvas.flush_events()
    plt.pause(0.001)


def keep_open(view: dict | None) -> None:
    """Leave the finished training on screen until the window is closed.

    Args:
        view: Output of open_view, or None.
    """
    if view is None:
        return
    print("Close the window to quit.")
    plt.ioff()
    plt.show(block=True)
