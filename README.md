# dslr (42cursus)

> A 42 Project: Implementation of Data Analysis, Data Visualization, and Multi-Class Logistic Regression (One-vs-Rest) from scratch.

---

## Table of Contents
- [1. Installation & Setup](#1-installation--setup)
- [2. How to Use](#2-how-to-use)
  - [Part 1: Data Analysis](#part-1-data-analysis)
  - [Part 2: Data Visualization](#part-2-data-visualization)
  - [Part 3: Logistic Regression](#part-3-logistic-regression)
  - [Part 4: Bonus](#part-4-bonus)

---

## 1. How to Use

### Part 1 : Data Analysis

describe.py

Recreated a custom statistical analysis function similar to pandas.describe() without using built-in statistical functions (mean, std, min, max, etc.).

```bash
python3 src/describe.py dataset_train.csv
```

### Part 2 : Data Visualization

histogram.py

scatter_plot.py

pair_plot.py

Scripts created to analyze features and make visual comparisons:

`histogram.py`: Visualizes course score distributions across all 4 houses to find the most homogeneous course.

```bash
python3 src/histogram.py datasets/dataset_train.csv
```

Question: Which Hogwarts course has a homogeneous score distribution between all four houses?

Response: Arithmancy and Care of Magical Creatures. The score distributions for all four houses overlap almost perfectly across the same range, meaning this feature does not help distinguish between houses (supported by a very low F-score of ~0.06).

`scatter_plot.py`: Plotting features against each other to find two similar features.

```bash
python3 src/scatter_plot.py datasets/dataset_train.csv
```

Question: What are the two features that are similar?

Response: History of Magic and Transfiguration (or Defense Against the Dark Arts and Astronomy). They display a linear relationship on the scatter plot, indicating high similarity and redundancy.

`pair_plot.py`: Displays a scatter plot matrix to decide which features to select for training.

```bash
python3 src/pair_plot.py datasets/dataset_train.csv
```

Question: From this visualization, which features are you going to use for your logistic regression?

Response: Based on the visualization and ANOVA F-scores, we select 10 features for training. We drop Arithmancy (homogeneous, no discriminatory value) and Astronomy (redundant with Defense Against the Dark Arts).

Selected features (Top 10):

    Defense Against the Dark Arts

    Astronomy

    Charms

    Ancient Runes

    Divination

    Herbology

    Transfiguration

    Muggle Studies

    Flying

    History of Magic


### Part 3 : Logistic Regression (In Progress / Work in Progress)

`logreg_train.py`: Trains the One-vs-Rest Logistic Regression model using Gradient Descent. Generates `weights.csv`.

```bash
python3 src/logreg_train.py datasets/dataset_train.csv
```

`logreg_predict.py`: Predicts Hogwarts Houses for the test dataset using the saved weights and outputs `houses.csv`.

```bash
python3 src/logreg_predict.py datasets/dataset_test.csv weights.csv
```

### Part 4 : Bonus (Pas encore)

1. Add more fields to describe.[extension]
2. Implement stochastic gradient descent
3. Implement other optimization algorithms (Batch GD, mini-batch GD, or others)
