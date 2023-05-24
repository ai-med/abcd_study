abcd_paper
==============================

Goal: Explore predictability of various psychiatric diagnoses based on neuroimaging features in subjects from the ABCD study.

Getting started
===============

1. Copy the following files into ``data/raw/``:

   From the baseline release of the ABCD study:
   ```
   abcd_ksad01.txt
   abcd_ksad501.txt
   acspsw03.txt
   btsv01.txt
   ```
   A table with FreeSurfer features (you need to run FreeSurfer on the sMRI data of the ABCD Study):
   ```
   abcd_freesurfer.csv
   ```
   A table with processed sociodemographic features. These features are the same as in the ABCD Neurocognitive Prediction Challenge. The respective R code can be found on the challenge website (https://sibis.sri.com/abcd-np-challenge/).
   ```
   sociodem_bl.csv
   ```

2. Run ``python src/runnable/make_dataset.py`` to process and combine these data into one dataframe. Use the following options:
   ```
   --select-one-child-per-family: Whether to randomly select only one child per family
   --seed: Random number seed for selecting one child per family
   ```
   In our article, a `seed` of 77 was used.

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
**Note:** Running these experiments will take extended amounts of time (about 20 hours for a single repeat of 5-fold cross validation on a fast machine). Consider parallelizing computations on several machines by using different seeds.

In our article, a `seed` of 77 was used.

Evaluation and visualization
============================

All raw predictions are saved to ``results/``.

Replication of published results
================================

We have provided a table (`data/splits.csv`) with the subject IDs in the training, validation, and test sets of each fold in our repeated cross validation scheme. You may use it to reproduce the results of our article *Can We Predict Mental Disorders in Children? A Large-Scale Assessment of Machine Learning on Structural Neuroimaging of 6916 Children in the ABCD Study*.

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
