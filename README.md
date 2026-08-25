*This project has been created as part of the 42 curriculum by mtian, pzinurov.*

# dslr (42cursus)

> Data Science x Logistic Regression: rebuild the Sorting Hat.

---

## Table of Contents

- [Description](#description)
- [Instructions](#instructions)
  - [Installation](#installation)
  - [Execution](#execution)
  - [Checking that it works](#checking-that-it-works)
- [Usage](#usage)
  - [Part 1 : Data Analysis](#part-1--data-analysis)
  - [Part 2 : Data Visualization](#part-2--data-visualization)
  - [Part 3 : Logistic Regression](#part-3--logistic-regression)
  - [Part 4 : Bonus](#part-4--bonus)
- [Project Layout](#project-layout)
- [Resources](#resources)

---

## Description

Professor McGonagall needs a replacement for the Sorting Hat. We are given a
dataset of Hogwarts students (their house, and their marks in 13 courses) and
we have to build a classifier that puts a new student in the right house.

The goal is to do the whole data science loop by hand, without calling the
library function that already solves it:

1. **Explore** the data. A `describe.py` that rebuilds `pandas.describe()`
   from scratch, with no `mean()`, `std()`, `min()`, `max()` or `percentile()`.
2. **Visualize** the data. Histograms, a scatter plot and a pair plot, used to
   decide which courses are actually worth training on.
3. **Train** a multi-class classifier. Four binary logistic regressions, one
   per house (One-vs-All), fitted with gradient descent written by hand.
4. **Predict** the house of every student in the test set.

The maths (percentiles, skewness, Z-score, sigmoid, log-loss, the gradient
update) are all implemented manually. `pandas` is used to read CSV files and
hold the tables, `matplotlib`/`seaborn` to draw, and `scikit-learn` only for
the ANOVA F-score used to rank features — never to fit the model.

Target: **98%+ accuracy** on the test set.

---

## Instructions

### Installation

Everything is installed in a local virtual environment. `prepare_env.sh`
activates it, so it has to be **sourced**, not executed:

```bash
source prepare_env.sh
```

That does three things:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

To leave the environment:

```bash
deactivate
```

### Execution

The training `logreg_train.py` should be run before the `logreg_predict.py`.

```bash
# 1. train  ->  writes model_params.json
python3 src/logreg_train.py datasets/dataset_train.csv

# 2. predict  ->  writes houses.csv
python3 src/logreg_predict.py datasets/dataset_test.csv
```

Both output files are written to the **current working directory**.
`model_params.json` is the only thing that carries state between
the two programs; it holds the selected features, the training means and
standard deviations, and the weights and bias of each house.

### Checking that it works

- **Training.** The accuracy panel of the animation should climb past the
  dashed 98% line. The last line printed is `All models trained successfully!`.
- **The model file.** `model_params.json` should exist and contain four
  houses under `weights`, each with 10 named weights and a bias:

  ```bash
  python3 -c "import json; d = json.load(open('model_params.json')); \
      print(list(d['weights']), len(d['best_features']))"
  ```

- **Predictions.** `houses.csv` should have one header line plus one line per
  student in the test set, and every house name should appear:

  ```bash
  wc -l houses.csv
  cut -d, -f2 houses.csv | sort | uniq -c
  ```

- **Style.** The same two linters that run in CI:

  ```bash
  flake8 .
  pydocstyle src
  ```

---

## Usage

### Part 1 : Data Analysis

`describe.py` — works similar to `pandas.describe()` on every numeric column.

```bash
python3 src/describe.py datasets/dataset_train.csv
```

It parses the CSV with the standard `csv` module, drops non-numeric columns,
and prints `Count`, `Missing`, `Mean`, `Std`, `Min`, `25%`, `50%`, `75%`,
`Max`, `Range`, `IQR` and `Skew`. Percentiles use linear interpolation, the
standard deviation is the sample one (`n - 1`), and the skewness carries the
usual sample-size correction.

`Missing`, `Range`, `IQR` and `Skew` are the bonus fields.

### Part 2 : Data Visualization

`histogram.py` — score distribution of every course, split by house.

```bash
python3 src/histogram.py datasets/dataset_train.csv
```

> **Question:** Which Hogwarts course has a homogeneous score distribution
> between all four houses?
>
> **Response:** Arithmancy and Care of Magical Creatures. The four curves sit
> on top of each other over the same range, so the course tells you nothing
> about which house a student belongs to. Their ANOVA F-scores are near zero,
> which says the same thing with a number.

`scatter_plot.py` — the two courses that look the most alike.

```bash
python3 src/scatter_plot.py datasets/dataset_train.csv
```

> **Question:** What are the two features that are similar?
>
> **Response:** Astronomy and Defense Against the Dark Arts. Their correlation
> is almost exactly -1: the distribution is on a straight line. One is a
> mirrored copy of the other, so keeping both adds no information.

`pair_plot.py` — scatter plot matrix of every course, coloured by house.

```bash
python3 src/pair_plot.py datasets/dataset_train.csv
```

> **Question:** From this visualization, which features are you going to use
> for your logistic regression?
>
> **Response:** The 10 courses with the highest F-score. `select_best_features()`
> computes the ranking and drops the three worst. The ranking is printed on every
> run, by `pair_plot.py` and by `logreg_train.py`.

### Part 3 : Logistic Regression

`logreg_train.py` — trains One-vs-All and saves the model.

```bash
python3 src/logreg_train.py datasets/dataset_train.csv [optimizer]
```

| optimizer | batch size | shuffle | note |
| --- | --- | --- | --- |
| `BGD` | whole set | no | default, mandatory part |
| `SGD` | 1 | yes | bonus |
| `minibatch` | 32 | yes | bonus |
| `minibatch-noshuffle` | 32 | no | to see what shuffling buys you |

The pipeline: fill missing marks with the column mean → rank the features and
keep the top 10 → Z-score normalise → train one binary classifier per house
with gradient descent → save everything to `model_params.json`. Learning rate
`0.1`, 300 epochs per house.

An animation window opens during training with four panels: the sigmoid with every
student placed on it, the log-loss per house, the learned weights, and the
running accuracy. It is throttled — early epochs are drawn often, later ones
rarely. Close the window to quit.

`logreg_predict.py` — predicts the houses of the test set.

```bash
python3 src/logreg_predict.py datasets/dataset_test.csv
```

Missing marks are filled with the *training* mean, the features are scaled with the
*training* mean and std, and each student gets the house whose sigmoid is
highest. Result is saved to `houses.csv`:

```
Index,Hogwarts House
0,Hufflepuff
1,Ravenclaw
...
```

### Part 4 : Bonus

- [x] More fields in `describe.py` — `Missing`, `Range`, `IQR`, `Skew`
- [x] Stochastic gradient descent — `SGD`
- [x] Other optimizers — `BGD`, `minibatch`, `minibatch-noshuffle`
- [x] Training animation — `training_plot.py`

---

## Project Layout

```
.
├── src/
│   ├── describe.py         # Part 1  - statistics from scratch
│   ├── histogram.py        # Part 2  - homogeneity
│   ├── scatter_plot.py     # Part 2  - correlation
│   ├── pair_plot.py        # Part 2  - matrix + feature selection
│   ├── logreg_train.py     # Part 3  - One-vs-All training
│   ├── logreg_predict.py   # Part 3  - prediction
│   ├── training_plot.py    # bonus   - live animation
│   └── utils.py            # shared  - validation, colors, short names
├── datasets/
│   ├── dataset_train.csv
│   └── dataset_test.csv
├── prepare_env.sh
├── requirements-dev.txt
├── setup.cfg               # flake8 + pydocstyle config
├── .github/workflows
│   └── lint.yml            # lint and style CI
```

Generated at runtime, not committed: `model_params.json`, `houses.csv`,
`.venv/`.

---

## Resources

**Logistic regression**

- [scikit-learn: Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
  — the reference implementation, read for comparison only.

**Statistics**

- [`pandas.DataFrame.describe`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)
  — the output `describe.py` had to match.
- [ANOVA F-test / `f_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html)
  — feature ranking.

**Plotting**

- [seaborn tutorial](https://seaborn.pydata.org/tutorial.html) — `histplot`,
  `scatterplot`, `pairplot`.
- [matplotlib event handling](https://matplotlib.org/stable/users/explain/figure/event_handling.html)
  — the scroll/drag zoom in `pair_plot.py`.
- [matplotlib interactive mode](https://matplotlib.org/stable/users/explain/interactive.html)
  — the live training window.

**Style**

- [PEP 8](https://peps.python.org/pep-0008/) and
  [PEP 257](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  — the docstring convention enforced by `pydocstyle`.
