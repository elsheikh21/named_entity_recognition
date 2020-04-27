import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import (confusion_matrix, precision_score,
                             precision_recall_fscore_support)
from tqdm.auto import tqdm


class Evaluator:
    def __init__(self, model, test_dataset, idx2label, is_crf):
        self.model = model
        self.test_dataset = test_dataset
        self.idx2label = idx2label
        self.is_crf = is_crf

    def compute_scores(self):
        all_predictions = list()
        all_labels = list()
        for step, samples in tqdm(enumerate(self.test_dataset), desc="Predicting batches of data"):
            inputs, labels = samples['inputs'], samples['outputs']
            if self.is_crf:
                predictions = self.model.predict(inputs)
                predictions = torch.LongTensor(predictions).to('cuda').view(-1)
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
        micro_precision_recall_f1score = precision_recall_fscore_support(all_labels, all_predictions,
                                                                         labels=list(range(len(self.idx2label))),
                                                                         average="micro")

        # precision per class and arithmetic average of them. Does not take into account class imbalance.
        macro_precision_recall_f1score = precision_recall_fscore_support(all_labels, all_predictions,
                                                                         labels=list(range(len(self.idx2label))),
                                                                         average="macro")

        per_class_precision = precision_score(all_labels, all_predictions,
                                              labels=list(range(len(self.idx2label))),
                                              average=None)

        confusion_mat = confusion_matrix(all_labels, all_predictions,
                                         labels=list(range(len(self.idx2label))),
                                         normalize='true')

        scores_dict = {"macro_precision_recall_f1score": macro_precision_recall_f1score,
                       "micro_precision_recall_f1score": micro_precision_recall_f1score,
                       "per_class_precision": per_class_precision,
                       "confusion_matrix": confusion_mat}

        return scores_dict

    def pprint_confusion_matrix(self, conf_matrix):
        df_cm = pd.DataFrame(conf_matrix)
        plt.figure(figsize=(10, 7))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
        save_to = os.path.join(os.getcwd(), "resources", f"{self.model.name}_confusion_matrix.png")
        plt.savefig(save_to)
        plt.show()

    def check_performance(self):
        scores = self.compute_scores()
        per_class_precision = scores["per_class_precision"]
        precision_, recall_, f1score_, _ = scores['macro_precision_recall_f1score']
        print(f"Macro Precision: {precision_}")
        print(f"Macro Recall: {recall_}")
        print(f"Macro F1_Score: {f1score_}")

        print("Per class Precision:")
        for idx_class, precision in sorted(enumerate(per_class_precision), key=lambda elem: -elem[1]):
            label = self.idx2label[idx_class]
            print(f'{label}: {precision}')

        precision, recall, f1score, _ = scores['micro_precision_recall_f1score']
        print(f"Micro Precision: {precision}")
        print(f"Micro Recall: {recall}")
        print(f"Micro F1_Score: {f1score}")

        confusion_matrix_ = scores['confusion_matrix']
        print(f'Confusion Matrix:\n {confusion_matrix_}')
        self.pprint_confusion_matrix(confusion_matrix_)
