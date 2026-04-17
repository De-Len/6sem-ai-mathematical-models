import numpy as np
from excel_viewer import ExcelViewer
from lab1.inference import inference_one_neuron
from lab1.mean_squared_error import mean_squared_error
from lab1.z_score import z_score

np.random.seed(42)
random_weights = np.random.uniform(-1, 1, 9)


def lab1():
    viewer = ExcelViewer("кредиты.xlsx")
    viewer.load()
    data = (viewer.get_head(900)
            .to_dict(orient="records"))

    # Подготовка данных
    X = []
    y_true = []

    for records in data:
        x_vect = []
        for key, value in records.items():
            if key == "Код":
                continue
            elif key != "Число просрочек более 60 дн.":
                x_vect.append(value)
            else:
                y_true.append(value)

        X.append(z_score(x_vect))

    X = np.array(X)
    y_true = np.array(y_true)

    # Градиентный спуск
    weights = np.random.uniform(-1, 1, 9)
    learning_rate = 0.01
    epochs = 1000

    for epoch in range(epochs):
        # Forward pass: предсказания для всех примеров
        predictions = np.array([inference_one_neuron(x, weights, 1) for x in X])

        # MSE loss
        loss = mean_squared_error(y_true, predictions)

        # Градиент: dLoss/dweights = (2/n) * X.T * (predictions - y_true)
        gradients = (2 / len(X)) * X.T @ (predictions - y_true)

        # Обновляем веса
        weights -= learning_rate * gradients

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, MSE: {loss:.6f}")

    print(f"\nФинальные веса: {weights}")
    print(f"Финальная MSE: {loss:.6f}")

    predictions = np.array([inference_one_neuron(x, weights, 1) for x in X])
    items_count = len(predictions)
    corrent_item = 0
    for i in range(items_count):
        if round(predictions[i]) == y_true[i]:
            corrent_item += 1

    print(f"% Правильных предсказаний: {(corrent_item / items_count) * 100}")




if __name__ == "__main__":
    lab1()