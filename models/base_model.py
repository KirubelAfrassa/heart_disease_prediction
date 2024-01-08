from abc import abstractmethod

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from data_pre_processor import data_normalizer
from util import separate_x_y
import seaborn as sns


class base_classifier:
    def __init__(self, train_data=None, test_data=None):
        self.predictions = None
        self.train_data, self.test_data = data_normalizer(train_data, test_data)
        self.X_train, self.y_train, self.X_test, self.y_test = separate_x_y([self.train_data, self.test_data],
                                                                            independent_vars=[col for col in
                                                                                              self.train_data if
                                                                                              col != "output"])

    @abstractmethod
    def classify(self):
        pass

    def plot(self):
        if self.predictions is None:
            raise SystemExit("Call classify() first before plot()")

        print('Output confusion matrix as a heat map: ')
        cm = confusion_matrix(y_pred=self.predictions, y_true=self.y_test)
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Expected')

        print("Classification Report:")
        print(classification_report(self.y_test, self.predictions))
