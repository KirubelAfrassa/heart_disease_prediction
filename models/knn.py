from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from models.base_model import base_classifier


class Knn(base_classifier):
    def __init__(self, train_data=None, test_data=None, find_best_params=False):
        super().__init__(train_data, test_data)

        if find_best_params:
            param_grid = {"n_neighbors": [5, 6, 7, 8, 9, 10], "p": [1, 2, 3, 4]}
            self.grid_search_cv = GridSearchCV(Knn, param_grid, cv=5, scoring="accuracy")
            self.grid_search_cv.fit(self.X_train, self.y_train)
            self.classifier = KNeighborsClassifier(**self.grid_search_cv.best_params_)
        else:
            self.classifier = KNeighborsClassifier(n_neighbors=5)

    def classify(self):
        self.classifier.fit(self.X_train, self.y_train)
        self.predictions = self.classifier.predict(self.X_test)

    def find_best_params(self):
        if self.grid_search_cv is None:
            print("Best Parameter Search is not Preformed While Creating Classifier.")
            return
        return self.grid_search_cv.best_params_

    def find_best_score(self):
        if self.grid_search_cv is None:
            print("Best Parameter Search is not Preformed While Creating Classifier.")
            return
        return self.grid_search_cv.best_score_
