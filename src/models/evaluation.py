from pathlib import Path
import json
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, log_loss


class ResultManager:
    """Utility object for computation and saving of results.

    Saves raw predictions, calculates and saves ROC
    AUC values, and writes to TensorBoard.

    Attributes:
        save_root (pathlib.Path): Root directory of current experiment run.
        roc_auc_values (dict): Dictionary holding ROC AUC values of each fold
        logger (pytorch_lightning.loggers.TensorBoardLogger): TensorBoard logger
            to write to.
    """
    def __init__(self,
                 tensorboard_logger,
                 save_root: Path,
                 save_params: dict = None):
        i = 0
        while (save_root / f"run_{i}").exists():
            i += 1
        self.save_root = save_root / f"run_{i}"
        self.save_root.mkdir(parents=True)
        # Save high level parameters (e.g. random_state)
        if save_params:
            with open(self.save_root / 'params.txt', 'w') as file:
                file.write(json.dumps(save_params))
        # Store ROC AUC values
        self.roc_auc_values = {}
        self.logger = tensorboard_logger

    def save_predictions(self,
                         dataset_name: str,
                         model_name: str,
                         fold: int,
                         split_set: str,
                         y_true: pd.DataFrame,
                         y_pred: pd.DataFrame) -> None:
        """Does two things:
         1. Saves raw predictions y_pred
         2. Computes ROC AUC values and stores them internally to save them
            later when finish() is called

        :param dataset_name: Name of full dataset (e.g. unpermuted, permuted_1,
            etc.)
        :param model_name: Name of model
        :param fold: Number of current fold in (repeated) k-fold CV
        :param split_set: Name of split set. Possible values: train, valid,
            test
        :param y_true: True labels of dataset
        :param y_pred: Predicted labels of dataset
        :return:
        """
        path = self.save_root / dataset_name / model_name / split_set
        if not path.exists():
            path.mkdir(parents=True)
        y_pred.to_csv(path / f"fold_{fold}.csv", index=True)
        # Compute and store ROC AUC value
        #   Set up ROC AUC dictionary
        if dataset_name not in self.roc_auc_values.keys():
            self.roc_auc_values[dataset_name] = {}
        if model_name not in self.roc_auc_values[dataset_name].keys():
            self.roc_auc_values[dataset_name][model_name] = {}
        if split_set not in self.roc_auc_values[dataset_name][model_name].keys():
            self.roc_auc_values[dataset_name][model_name][split_set] = {}
        #   Compute ROC AUC values
        mlbe = MultilabelBinaryEvaluator(y_true=y_true, y_pred=y_pred)
        self.roc_auc_values[dataset_name][model_name][split_set][fold] = mlbe.roc_auc()
        #   Write to TensorBoard
        for key in self.roc_auc_values[dataset_name][model_name][split_set][fold].keys():
            # Track ROC AUC values
            self.logger.experiment.add_scalar(
                f"{dataset_name}/{model_name}/{split_set}/ROC_AUC_mean_{key}",
                self.roc_auc_values[dataset_name][model_name][split_set][fold][key],
                global_step=fold
            )
            # Track histogram of ROC AUC values
            roc_auc_list = []
            for idx in self.roc_auc_values[dataset_name][model_name][split_set].keys():
                roc_auc_list.append(
                    self.roc_auc_values[dataset_name][model_name][split_set][idx][key]
                )
            self.logger.experiment.add_histogram(
                f"{dataset_name}/{model_name}/{split_set}/ROC_AUC_{key}",
                np.array(roc_auc_list),
                global_step=fold
            )

    def finish(self) -> None:
        """
        Saves ROC AUC values as dataframes. Call this function after training
        is finished.
        """
        for dataset_name, item0 in self.roc_auc_values.items():
            for model_name, item1 in item0.items():
                for split_set, item2 in item1.items():
                    path = self.save_root / dataset_name / model_name / split_set
                    roc_auc_df = pd.DataFrame.from_dict(item2, orient='index')
                    roc_auc_df.to_csv(path / 'roc_auc.csv', index=True)


class MultilabelBinaryEvaluator:

    def __init__(self,
                 y_true: pd.DataFrame,
                 y_pred: pd.DataFrame):
        if y_true.shape != y_pred.shape:
            raise Exception("Error: y_true and y_pred are not of equal shape!")
        if np.isnan(y_true).any().any():
            raise Exception("Error: y_true contains NaN cells!")
        if np.isnan(y_pred).any().any():
            raise Exception("Error: y_pred contains NaN cells!")

        self._y_true = y_true
        self._y_pred = y_pred
        self.per_label_roc_auc = None

    def roc_auc(self):
        """Compute area under ROC curve"""
        if self.per_label_roc_auc is None:
            self.per_label_fpr = {}
            self.per_label_tpr = {}
            self.per_label_thresholds_roc = {}

            self.per_label_roc_auc = {}
            for col in self._y_true.columns:
                self.per_label_fpr[col], self.per_label_tpr[col], \
                self.per_label_thresholds_roc[col] = roc_curve(
                    y_true=self._y_true[col],
                    y_score=self._y_pred[col]
                )
                self.per_label_roc_auc[col] = auc(
                    self.per_label_fpr[col], self.per_label_tpr[col]
                )

        return self.per_label_roc_auc

    def compute_pr(self, kind: str):
        """Compute precision, recall and thresholds for precision recall curve"""
        if kind not in ['pooled', 'per_label']:
            raise Exception("'kind' argument must be one out of ['pooled', 'per_label']!")

        if kind == 'pooled':
            self.pooled_precision, self.pooled_recall, self.pooled_thresholds_pr = precision_recall_curve(
                y_true=self.y_true.flatten() > 0.5,
                probas_pred=self.y_pred.flatten()
            )
        elif kind == 'per_label':
            self.per_label_precision = []
            self.per_label_recall = []
            self.per_label_thresholds_pr = []
            for label in range(self.y_true.shape[1]):
                label_precision, label_recall, label_thresholds_pr = precision_recall_curve(
                    y_true=self.y_true[:, label] > 0.5,
                    probas_pred=self.y_pred[:, label]
                )
                self.per_label_precision.append(label_precision)
                self.per_label_recall.append(label_recall)
                self.per_label_thresholds_pr.append(label_thresholds_pr)

    def pr_auc(self, kind: str):
        """Compute area under precision recall curve"""
        if kind not in ['pooled', 'per_label']:
            raise Exception("'kind' argument must be one out of ['pooled', 'per_label']!")

        if kind == 'pooled':
            return auc(self.pooled_recall, self.pooled_precision)
        elif kind == 'per_label':
            per_label_pr_auc = []
            for label in range(self.y_true.shape[1]):
                per_label_pr_auc.append(
                    auc(self.per_label_recall[label], self.per_label_precision[label])
                )
            return per_label_pr_auc

    def pr_auc_adj(self, kind: str):
        """AUC above horizontal random predictor baseline, devided by total area above
        horizontal random predictor baseline. This creates a precision recall AUC normalized
        to 1 (negative values indicate performance worse than majority class classifier).
        Will be used to compare between different experiments."""
        if kind not in ['pooled', 'per_label']:
            raise Exception("'kind' argument must be one out of ['pooled', 'per_label']!")

        if kind == 'pooled':
            return (self.pr_auc('pooled') - self.pooled_precision[0]) / (
                        1 - self.pooled_precision[0])
        elif kind == 'per_label':
            per_label_pr_auc_adj = []
            for _pr_auc, _prec in zip(self.pr_auc('per_label'), self.per_label_precision):
                per_label_pr_auc_adj.append(
                    (_pr_auc - _prec[0]) / (1 - _prec[0])
                )
            return per_label_pr_auc_adj

    def plot_pr_roc_curves(self, kind: str, titletext: str = None):
        """
        Show precision-recall and ROC curve.

        Parameters
        ==========

        kind : str, in ['pooled', 'per_label']
            How ROC and PR curves should be created. Pool all predictions together and
            generate one curve or generate one separate curve for each label?

        titletext : str, optional
            String for plot suptitle

        Return value
        ============

        no return value
        """

        if kind not in ['pooled', 'per_label']:
            raise Exception("'kind' argument must be one out of ['pooled', 'per_label']!")

        self.compute_roc(kind=kind)
        self.compute_pr(kind=kind)

        linewidth = 1
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(t=titletext, size=18, y=.98, va='bottom')

        if kind == 'pooled':
            color = 'darkorange'
            # Precision-Recall curve
            axes[0].plot(
                self.pooled_recall,
                self.pooled_precision,
                lw=linewidth,
                # label="Corrected AUC = %0.3f" % self.pr_auc_corrected,
                color=color
            )
            # Plot naive baseline in precision-recall plot
            baseline_precision = self.pooled_precision[0]
            axes[0].plot(
                [0, 1], [baseline_precision, baseline_precision], color='navy', lw=linewidth,
                linestyle='--'
            )
            # ROC
            axes[1].plot(
                self.pooled_fpr,
                self.pooled_tpr,
                lw=linewidth,
                # label="AUC = %0.3f" % self.roc_auc,
                color=color
            )
        elif kind == 'per_label':
            for label in range(self.y_true.shape[1]):
                # Precision-Recall curve
                axes[0].plot(
                    self.per_label_recall[label],
                    self.per_label_precision[label],
                    lw=linewidth,
                    # label="Corrected AUC = %0.3f" % self.pr_auc_corrected,
                )
                # ROC
                axes[1].plot(
                    self.per_label_fpr[label],
                    self.per_label_tpr[label],
                    lw=linewidth,
                    # label="AUC = %0.3f" % self.roc_auc,
                )

        axes[0].set_xlim([0.0, 1.05])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].tick_params(labelbottom=False, labelleft=False)
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision-Recall curve', size=12)
        # axes[0].legend(loc='upper right')

        axes[1].plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--')
        axes[1].set_xlim([0.0, 1.05])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].tick_params(labelbottom=False, labelleft=False)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC', size=12)
        # axes[1].legend(loc='lower right')

        plt.show()


class BinaryEvaluator:
    """Evaluates predictions of binary classifiers and stores performance data.

    Parameters
    ==========

    y_true : array-like
        True labels

    y_pred : array-like
        Predictions

    precision : list of float
        Precision scores at different thresholds

    recall : list of float
        Recall scores at different thresholds

    thresholds_prc : list of float
        Thresholds for precision and recall

    tpr : list of float
        True positive rates (sensitivity) at different thresholds

    fpr : list of float
        False positive rates (1 - specificity) at different thresholds

    thresholds_roc : list of float
        Thresholds for tpr and fpr

    binary_cross_entropy : float
        Binary cross entropy measure
    """

    def __init__(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise Exception("y_true and y_pred are not of equal length!")

        self.y_true = y_true
        self.y_pred = y_pred
        self.precision = None
        self.recall = None
        self.thresholds_prc = None
        self.tpr = None
        self.fpr = None
        self.thresholds_roc = None
        self.binary_cross_entropy = None

        self.evaluate()

    @property
    def roc_auc(self) -> float:
        """Area under ROC curve"""
        return auc(self.fpr, self.tpr)

    @property
    def pr_auc(self) -> float:
        """Area under precision recall curve"""
        return auc(self.recall, self.precision)

    @property
    def pr_auc_corrected(self) -> float:
        """AUC above horizontal random predictor baseline, devided by total area above
        horizontal random predictor baseline. This creates a precision recall AUC normalized
        to 1 (negative values indicate performance worse than majority class classifier).
        Will be used to compare between different experiments."""
        return (self.pr_auc - self.precision[0]) / (1 - self.precision[0])

    def evaluate(self) -> None:
        """Make performance scores"""
        self.precision, self.recall, self.thresholds_prc = precision_recall_curve(
            y_true=self.y_true > 0.5, probas_pred=self.y_pred
        )
        self.fpr, self.tpr, self.thresholds_roc = roc_curve(
            y_true=self.y_true > 0.5, y_score=self.y_pred
        )
        self.binary_cross_entropy = log_loss(
            self.y_true.astype(int), self.y_pred, labels=[0, 1]
        )

    """def plot_y_pred_distribution(self, ax:plt.axis, title:str, hist:bool=True, kde:bool=False):
        Show distributions of predictions between sample groups y_true = 0 and y_true = 1
        sns.distplot(
            y_data_nona.loc[y_data_nona['y_true'] == 0, 'y_pred'],
            label='y_pred | y_true = 0',
            ax=ax,
            hist=hist,
            kde=kde
        )
        sns.distplot(
            y_data_nona.loc[y_data_nona['y_true'] == 1, 'y_pred'],
            label='y_pred | y_true = 1',
            ax=ax,
            hist=hist,
            kde=kde
        )
        ax.set_xlim((0, 1))
        ax.legend()
        ax.set_title(title, size=12)"""

    def plot_pr_roc_curves(self,
                           titletext: str = None) -> None:
        """
        Show precision-recall and ROC curve.

        Parameters
        ==========

        titletext : str, optional
            String for plot suptitle

        Return value
        ============

        no return value
        """

        color = 'darkorange'
        linewidth = 1

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        fig.suptitle(t=titletext, size=18, y=.98, va='bottom')

        # Precision-Recall curve
        axes[0].plot(
            self.recall,
            self.precision,
            lw=linewidth,
            label="Corrected AUC = %0.3f" % self.pr_auc_corrected,
            color=color
        )
        axes[0].set_xlim([0.0, 1.05])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].tick_params(labelbottom=False, labelleft=False)
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision-Recall curve', size=12)
        axes[0].legend(loc='upper right')

        # ROC
        axes[1].plot(
            self.fpr,
            self.tpr,
            lw=linewidth,
            label="AUC = %0.3f" % self.roc_auc,
            color=color
        )
        axes[1].plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--')
        axes[1].set_xlim([0.0, 1.05])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].tick_params(labelbottom=False, labelleft=False)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC', size=12)
        axes[1].legend(loc='lower right')

        # Plot naive baseline in precision-recall plot
        baseline_precision = self.precision[0]
        axes[0].plot(
            [0, 1], [baseline_precision, baseline_precision], color='navy', lw=linewidth,
            linestyle='--'
        )
        plt.show()
