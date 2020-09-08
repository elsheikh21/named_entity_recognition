import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import (confusion_matrix, precision_score,
                             precision_recall_fscore_support,
                             recall_score, f1_score)
from tqdm.auto import tqdm
from typing import List, Any


def flat_list(l: List[List[Any]]) -> List[Any]:
    return [_e for e in l for _e in e]


class Evaluator:
    def __init__(self, model, test_dataset, is_crf):
        """
        Responsible for check model performance metrics, plotting of confusion matrix
        Args:
            model: nn.Module
            test_dataset: TSVDatasetParser Object
            is_crf: Flag, to imply approach for predictions
        """
        self.model = model
        self.test_dataset = test_dataset
        self.is_crf = is_crf
        self.micro_scores = None
        self.macro_scores = None
        self.class_scores = None
        self.confusion_matrix = None

    def compute_scores(self):
        """
        Fetches model's predictions, then computes performance by measuring macro and micro different metrics
        (Precision, Recall, F1Score), as well as, confusion matrix,
        Returns:

        """
        all_predictions = list()
        all_labels = list()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for step, samples in tqdm(enumerate(self.test_dataset), desc="Predicting batches of data"):
            inputs, labels = samples['inputs'], samples['outputs']
            if self.is_crf:
                predictions = self.model.predict(inputs)
                # print(flat_list(predictions))
                predictions = torch.LongTensor(predictions).to(device).view(-1)
            else:
                predictions = self.model(inputs)
                predictions = torch.argmax(predictions, -1).view(-1)
            labels = labels.view(-1)
            valid_indices = labels != 0
            valid_predictions = predictions[valid_indices]
            valid_labels = labels[valid_indices]
            all_predictions.extend(valid_predictions.tolist())
            all_labels.extend(valid_labels.tolist())
        # global precision. Does take class imbalance into account.
        self.micro_scores = precision_recall_fscore_support(all_labels, all_predictions, zero_division=0,
                                                            average="micro")

        # precision per class and arithmetic average of them. Does not take into account class imbalance.
        self.macro_scores = precision_recall_fscore_support(all_labels, all_predictions, zero_division=0,
                                                            average="macro")

        self.class_scores = precision_score(all_labels, all_predictions, zero_division=0,
                                            average=None)

        self.confusion_matrix = confusion_matrix(all_labels, all_predictions,
                                                 normalize='true')
        p = precision_score(all_labels, all_predictions, average='macro')
        r = recall_score(all_labels, all_predictions, average='macro')
        f = f1_score(all_labels, all_predictions, average='macro')
        print("=" * 30)
        print(f'Macro Precision: {p:0.4f}, Macro Recall: {r:0.4f}, Macro F1 Score: {f:0.4f}')


    def pprint_confusion_matrix(self, conf_matrix):
        """
        Plots Confusion matrix heat map
        Args:
            conf_matrix:

        Returns:
            None
        """
        df_cm = pd.DataFrame(conf_matrix)
        fig = plt.figure(figsize=(10, 7))
        axes = fig.add_subplot(111)
        sn.set(font_scale=1.5)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, ax=axes)  # font size
        axes.set_xlabel('Predicted labels')
        axes.set_ylabel('Labels')
        axes.set_title('Confusion Matrix')
        axes.xaxis.set_ticklabels(['PER', 'ORG', 'LOC', 'O'])
        axes.yaxis.set_ticklabels(['PER', 'ORG', 'LOC', 'O'])
        plt.savefig('./best_model.png')
        plt.show()

    def check_performance(self, idx2label):
        """
        invoke compute scores, then print results
        Args:
            idx2label: dict

        Returns:

        """
        self.compute_scores()
        precision_, recall_, f1score_, _ = self.macro_scores
        print("=" * 30)
        print(f"Macro Precision: {precision_}")
        print(f"Macro Recall: {recall_}")
        print(f"Macro F1_Score: {f1score_}")

        print("=" * 30)
        print("Per class Precision:")
        for idx_class, precision in sorted(enumerate(self.class_scores, start=1), key=lambda elem: -elem[1]):
            label = idx2label[idx_class]
            print(f'{label}: {precision}')

        print("=" * 30)
        precision, recall, f1score, _ = self.micro_scores
        print(f"Micro Precision: {precision}")
        print(f"Micro Recall: {recall}")
        print(f"Micro F1_Score: {f1score}")
        print("=" * 30)

        self.pprint_confusion_matrix(self.confusion_matrix)
