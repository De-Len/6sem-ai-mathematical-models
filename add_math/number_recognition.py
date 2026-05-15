import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    # ===================== 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ =====================
    print("1. Загрузка данных MNIST...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(f"   Обучающая выборка: {X_train.shape}")
    print(f"   Тестовая выборка: {X_test.shape}")

    # Нормализация: приводим пиксели от 0 до 1
    X_train = X_train.reshape(-1, 784).astype("float32") / 255.0
    X_test = X_test.reshape(-1, 784).astype("float32") / 255.0

    # One-hot encoding для меток
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    print(f"   После обработки X_train: {X_train.shape}")
    print(f"   После обработки y_train: {y_train.shape}")

    # ===================== 2. СОЗДАНИЕ МОДЕЛИ НЕЙРОСЕТИ =====================
    print("\n2. Создание модели нейросети...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,)),  # Скрытый слой 1
        Dropout(0.2),  # Dropout для предотвращения переобучения
        Dense(64, activation='relu'),  # Скрытый слой 2
        Dropout(0.2),
        Dense(10, activation='softmax')  # Выходной слой (10 цифр)
    ])

    # Компиляция модели
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # ===================== 3. ОБУЧЕНИЕ МОДЕЛИ =====================
    print("\n3. Обучение модели...")
    history = model.fit(
        X_train, y_train,
        batch_size=128,  # Размер пакета
        epochs=2,  # Количество эпох
        validation_split=0.1,  # 10% данных для валидации
        verbose=1  # Показывать прогресс
    )

    # ===================== 4. ОЦЕНКА НА ТЕСТОВЫХ ДАННЫХ =====================
    print("\n4. Оценка модели на тестовых данных...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Точность на тестовых данных: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"   Потери на тестовых данных: {test_loss:.4f}")

    # ===================== 5. ИНФЕРЕНС (ПРЕДСКАЗАНИЕ) =====================
    print("\n5. Инференс на нескольких примерах из тестовой выборки...")

    # Выбираем 5 случайных примеров из тестовой выборки
    num_samples = 5
    indices = np.random.choice(len(X_test), num_samples, replace=False)

    # Восстанавливаем исходную форму для отображения
    X_test_original = X_test.reshape(-1, 28, 28)

    for i, idx in enumerate(indices):
        # Получаем предсказание
        sample = X_test[idx].reshape(1, -1)
        prediction = model.predict(sample, verbose=0)
        predicted_class = np.argmax(prediction)
        true_class = np.argmax(y_test[idx])

        # Отображаем изображение
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X_test_original[idx], cmap='gray')
        plt.title(f'True: {true_class}\nPred: {predicted_class}')
        plt.axis('off')
        plt.tight_layout()

    plt.show()

    # ===================== 6. ГРАФИКИ ОБУЧЕНИЯ =====================
    print("\n6. Построение графиков обучения...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # График точности
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # График потерь
    axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='o')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # ===================== 7. СОХРАНЕНИЕ МОДЕЛИ (опционально) =====================
    print("\n7. Сохранение модели...")
    model.save('mnist_model.keras')
    print("   Модель сохранена как 'mnist_model.keras'")

    # ===================== 8. ЗАГРУЗКА И ИСПОЛЬЗОВАНИЕ СОХРАНЁННОЙ МОДЕЛИ =====================
    print("\n8. Пример загрузки сохранённой модели и инференса...")
    from tensorflow.keras.models import load_model

    # Загружаем модель
    loaded_model = load_model('mnist_model.keras')

    # Тестируем загруженную модель
    sample = X_test[0].reshape(1, -1)
    prediction = loaded_model.predict(sample, verbose=0)
    predicted_class = np.argmax(prediction)
    true_class = np.argmax(y_test[0])

    print(f"   Загруженная модель предсказала: {predicted_class}, правильный ответ: {true_class}")

    # ===================== 9. ФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ ПОЛЬЗОВАТЕЛЬСКИХ ИЗОБРАЖЕНИЙ =====================
    print("\n9. Функция для предсказания на новых изображениях:")


    def predict_digit(image_array, model):
        """
        Предсказывает цифру на изображении 28x28
        image_array: numpy array формы (28, 28) или (784,)
        """
        if image_array.shape == (28, 28):
            image_array = image_array.reshape(1, -1)
        elif image_array.shape == (784,):
            image_array = image_array.reshape(1, -1)

        image_array = image_array.astype("float32") / 255.0
        prediction = model.predict(image_array, verbose=0)
        return np.argmax(prediction), np.max(prediction) * 100


    # Тестируем функцию
    test_idx = 42
    test_image = X_test_original[test_idx]
    pred_digit, confidence = predict_digit(test_image, loaded_model)
    print(f"   Тест функции на примере {test_idx}:")
    print(f"   Предсказанная цифра: {pred_digit} с уверенностью {confidence:.2f}%")
    print(f"   Правильная цифра: {np.argmax(y_test[test_idx])}")

    print("\n✅ Обучение и тестирование успешно завершены!")