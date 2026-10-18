from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate_model(model, x_cal, y_cal):
    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(model), method="sigmoid")
    calibrated_model.fit(x_cal, y_cal)
    return calibrated_model
