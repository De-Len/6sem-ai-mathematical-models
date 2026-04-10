import numpy as np

global_epsilon = 0.000000001


def differentiable_function(x):
    """Пример функции, для которой считаем производную"""
    return x ** 2 + 2 * x + 1  # Производная: 2x + 2


def derivative_central(f, x, h=1e-7):
    """
    Центральная разность для вычисления производной
    f: функция
    x: точка, в которой вычисляем производную
    h: шаг
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def derivative_y(f, start=1, end=3000, epsilon=global_epsilon):
    """
    Вычисление производной в диапазоне точек
    """
    summ = 0
    points = []
    derivatives = []

    for i in range(start, end):
        x = i * epsilon
        der = derivative_central(f, x)
        points.append(x)
        derivatives.append(der)
        summ += der

    return summ, points, derivatives


# Пример использования:
if __name__ == "__main__":
    # Для одной точки:
    x_point = 2.5
    derivative_at_point = derivative_central(differentiable_function, x_point)
    print(f"Производная в точке x={x_point}: {derivative_at_point}")

    # Для диапазона точек:
    total_sum, x_points, derivs = derivative_y(differentiable_function)
    print(f"Сумма производных: {total_sum}")
    print(f"Первые 5 значений: {list(zip(x_points[:5], derivs[:5]))}")