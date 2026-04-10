import numpy as np

from excel_viewer import ExcelViewer
from lab1.derivative_y import derivative_central
from lab1.inference import inference_one_neuron
from lab1.mean_squared_error import mean_squared_error
from lab1.z_score import z_score

np.random.seed(42)  # для воспроизводимости
random_weights = np.random.uniform(-1, 1, 9)

# test_data = np.linspace(-3, 3, 9)


def lab1():
    viewer = ExcelViewer("кредиты.xlsx")
    viewer.load()

    data = (viewer.get_head(900)
            .to_dict(orient="records"))

    y_list_predict = []
    y_list_correct = []

    for records in data:
        print(records)
        x_vect = []
        y_correct = ""
        for key, value in records.items():
            if key == "Код":
                continue
            elif key != "Число просрочек более 60 дн.":
                x_vect.append(value)
            else:
                y_correct = value
                y_list_correct.append(y_correct)

        x_vect_with_z_score = z_score(x_vect)
        print(x_vect_with_z_score)

        y_list_predict.append(inference_one_neuron(x_vect_with_z_score, random_weights, 1))

    mse = mean_squared_error(y_list_correct, y_list_predict)
    print(f"MSE: {mse}")

    # Вычисление производной для loss функции (если нужно)
    # Например, производная MSE по предсказаниям:
    def loss_function(pred):
        return mean_squared_error(y_list_correct, pred)

    # Вычисляем производную в текущей точке
    current_predictions = np.array(y_list_predict)
    h = 1e-7
    derivative_loss = derivative_central(
        lambda p: mean_squared_error(y_list_correct, p.tolist()),
        current_predictions[0]  # для первого элемента
    )
    print(f"Производная MSE: {derivative_loss}")


if __name__ == "__main__":
    lab1()