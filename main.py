import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 1) # Первый слой
        self.fc2 = nn.Linear(1, 1) # Второй слой

    def forward(self, x): # Функция для инференса
        x = self.fc1(x) # Сначала проходим 1 слой
        x = self.fc2(x) # Потом второй
        return x

def main():
    x = [
        [1, 1],
        [2, 1],
        [3, 1],
        [4, 1],
        [5, 1],
        [1, 2],
        [2, 2],
        [3, 2],
        [4, 2],
        [5, 2],
        [1, 5]
    ]

    y = [2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 6]

    x_tensor = (torch.tensor(x)
                .float()) # Нейросеть работает только с float, а в массивах заданы int
    y_tensor = (torch.tensor(y)
                .float()
                .unsqueeze(1) # Добавляем размерность (для совпадения рангов тензоров)
                )

    net = SimpleNN() # Создание нейросети

    criterion = nn.MSELoss() # Функция потерь (среднеквадратичная ошибка)
    optimizer = optim.SGD(net.parameters(), lr=0.01) # Стохастический градиентный спуск (на все параметры, 0.01 - скорость обучения, шаг обновления весов)

    epochs = 10000
    for epoch in range(epochs):
        outputs = net(x_tensor) # Вводим x тензор и получаем вывод
        loss = criterion(outputs, y_tensor) # вычисляем функцию потерь

        optimizer.zero_grad() # очищаем градиенты (сначала их нет, но потом будут)

        # Метод обратного распространения ошибки (да?)
        loss.backward() # Считаем новые градиенты

        optimizer.step() # Сохраняем


        # Выводим в консоль для понимания (каждый 20 итераций)
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.8f}')


    with torch.no_grad(): # Контекстный менеджер. Убираем вычисление градиентов (для экономии). Короче неважно
        test_input = torch.tensor([[3.00003, 1]], dtype=torch.float32) # Вводим: 3, 2
        output = net(test_input)
        print(output) # Вывод

if __name__ == "__main__":
    main()