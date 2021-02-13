abcd_paper
==============================

Goal: Explore predictability of various psychiatric diagnoses based on neuroimaging features in subjects from the ABCD study.

Getting started
===============

1. Create a new directory ``data/raw/`` in the root of this repository and copy the following files from the baseline release of the ABCD study into it:

   ```
   abcd_ksad01.txt
   abcd_ksad501.txt
   acspsw03.txt
   btsv01.txt
   ```

2. Run ``python src/runnable/make_dataset.py`` to process and combine these data into one dataframe.

Running the experiments
=======================

1. To fit and obtain training, validation, and test set predictions by the OVR logistic regression, CCE logistic regression, and CCE Bayesian optimized XGBoost models on the processed dataset, run ``python src/runnable/run_unpermuted.py``. Use the following options:
    ```
    --seed: Random number seed (int)
    --k: Number of cross validation folds (int, default 5)
    --n: Number of successive k-fold CV runs (int)
    ```
2. To fit and obtain predictions on random permutations of the processed dataset, run ``python src/runnable/run_permuted.py`` using the following options:
   ```
   --seed: Random number seed (int)
   --k: Number of cross validation folds (int, default 5)
   --n: Number of successive k-fold CV runs (int)
   --num_permutations: Number of random permutations (int)
   ```
**Note:** Running these experiments will take extended amounts of time (about 20 hours for a single repeat of 5-fold cross validation on a fast machine) and consider parallelizing computations on several machines by using different seeds.

Evaluation and visualization
============================

All raw predictions are saved to ``results/``.

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
