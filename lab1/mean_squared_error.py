import numpy as np


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error (Среднеквадратичная ошибка)

    Args:
        y_true: реальные значения
        y_pred: предсказанные значения

    Returns:
        float: MSE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean((y_true - y_pred) ** 2)
