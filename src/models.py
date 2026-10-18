from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


class LabelEncodedXGBClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, parameters=None, random_state=None):
        self.parameters = parameters
        self.random_state = random_state

    def fit(self, x, y):
        self.label_encoder_ = LabelEncoder().fit(y)
        encoded_y = self.label_encoder_.transform(y)
        model_parameters = dict(self.parameters or {})
        if self.random_state is not None:
            model_parameters["random_state"] = self.random_state
        self.model_ = XGBClassifier(**model_parameters)
        self.model_.fit(x, encoded_y)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict_proba(self, x):
        return self.model_.predict_proba(x)

    def predict(self, x):
        encoded_predictions = self.model_.predict(x).astype(int)
        return self.label_encoder_.inverse_transform(encoded_predictions)


MODEL_TYPES = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "linear_svm": LinearSVC,
    "xgboost": LabelEncodedXGBClassifier
}


def build_model_factory(model_type, parameters):
    if model_type is LabelEncodedXGBClassifier:
        return lambda random_state: LabelEncodedXGBClassifier(parameters, random_state)

    def make_model(random_state):
        model_parameters = dict(parameters)
        model_parameters["random_state"] = random_state
        return model_type(**model_parameters)

    return make_model
