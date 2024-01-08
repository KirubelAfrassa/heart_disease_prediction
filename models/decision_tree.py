from keras.losses import poisson
from matplotlib import pyplot as plt
from numpy.random import uniform
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

from models.base_model import base_classifier


class Decision_tree(base_classifier):
    def __init__(self, train_data=None, test_data=None, find_best_params=False):
        super().__init__(train_data, test_data)

        if find_best_params:
            parameters = {'max_depth': poisson(mu=2, loc=2),
                          'max_leaf_nodes': poisson(mu=5, loc=5),
                          'min_samples_split': uniform(),
                          'min_samples_leaf': uniform()}
            self.random_search_cv = RandomizedSearchCV(DecisionTreeClassifier(random_state=28),
                                                       parameters, cv=5, n_iter=100, random_state=28)
            self.random_search_cv.fit(self.X_train, self.y_train)
            dt_params = self.random_search_cv.best_params_
            dt_params['min_samples_split'] = np.ceil(dt_params['min_samples_split'] * self.X_train.shape[0])
            dt_params['min_samples_leaf'] = np.ceil(dt_params['min_samples_leaf'] * self.X_train.shape[0])

            self.dt_params = dt_params
            self.classifier = self.random_search_cv.best_estimator_
        else:
            self.classifier = DecisionTreeClassifier()

    def classify(self):
        self.classifier.fit(self.X_train, self.y_train)
        self.predictions = self.classifier.predict(self.X_test)

    def plot_decision_tree(self):
        print('Plotting decision tree...')
        fig = plt.figure(figsize=(16, 8))
        _ = plot_tree(self.classifier,
                      feature_names=[col for col in
                                     self.train_data if
                                     col != "output"],
                      filled=True,
                      class_names=["1", "0"],
                      fontsize=10)

    def find_best_params(self):
        return self.dt_params
