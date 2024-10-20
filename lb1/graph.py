import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns


def analyze_sequence(sequence):
    # Преобразуем последовательность в массив NumPy
    sequence = np.array(sequence)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(sequence, marker="o", linestyle="-", color="b")
    plt.title("График значений последовательности")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.grid()
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")  # Добавление линии y=0
    plt.show()

    # Определяем характер последовательности
    is_increasing = np.all(np.diff(sequence) > 0)
    is_decreasing = np.all(np.diff(sequence) < 0)

    if is_increasing:
        print("Последовательность является возрастающей.")
    elif is_decreasing:
        print("Последовательность является убывающей.")
    else:
        print("Последовательность не является строго возрастающей или убывающей.")

        # Проверка на периодичность
        period = None
        for p in range(1, len(sequence) // 2 + 1):
            if np.array_equal(sequence[:-p], sequence[p : len(sequence) - p]):
                period = p
                break

        if period:
            print(
                f"Последовательность является периодической с длиной периода: {period}."
            )
        else:
            print("Последовательность не является периодической.")


def analyze_sequences(sequence1, sequence2):
    # Преобразуем последовательности в массивы NumPy
    sequence1 = np.array(sequence1)
    sequence2 = np.array(sequence2)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(sequence1, marker="o", linestyle="-", color="b", label="Исходная выборка")
    plt.plot(
        sequence2,
        marker="x",
        linestyle="--",
        color="r",
        label="Сгенерировванная выборка",
    )
    plt.title("График значений двух последовательностей")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.grid()
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")  # Линия y=0
    plt.legend()  # Добавление легенды для различения массивов
    plt.show()


def autocorrelation_analysis(sequence, max_lag=10):
    # Преобразуем последовательность в массив NumPy
    sequence = np.array(sequence)

    # Вычисляем среднее значение
    mean = np.mean(sequence)
    n = len(sequence)

    # Инициализируем список для хранения коэффициентов автокорреляции
    autocorr_values = []

    for lag in range(1, max_lag + 1):  # Начинаем с 1
        # Вычисляем автоковариацию
        autocov = np.sum((sequence[:-lag] - mean) * (sequence[lag:] - mean)) / n
        # Вычисляем автокорреляцию
        autocorr = autocov / np.var(sequence)
        autocorr_values.append(autocorr)

    # Создаем DataFrame для красивого отображения
    df = pd.DataFrame(
        {
            "Сдвиг": range(1, max_lag + 1),  # Только положительные сдвиги
            "Коэффициент автокорреляции": autocorr_values,
        }
    )

    # Выводим таблицу
    print(df.to_string(index=False))

    # Проверка на случайность
    significant_lags = [
        lag
        for lag, value in zip(range(1, max_lag + 1), autocorr_values)
        if abs(value) > 0.2
    ]

    if len(significant_lags) > 0:
        print(
            f"Последовательность не является случайной, обнаружены значимые автокорреляции на сдвиге(ах): {significant_lags}"
        )
    else:
        print("Последовательность может считаться случайной.")


def plot_frequency_histogram(data, bins=20):

    plt.figure(figsize=(10, 6))  # Размер графика
    plt.hist(data, bins=bins, edgecolor="black", alpha=0.7)  # Построение гистограммы
    plt.title("Гистограмма распределения частот")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.grid(axis="y", alpha=0.75)  # Сетка для лучшего восприятия
    plt.show()


def plot_frequency_histograms(data1, data2, bins=20):
    plt.figure(figsize=(10, 6))  # Размер графика

    # Построение двух гистограмм на одном графике
    plt.hist(
        data1,
        bins=bins,
        edgecolor="black",
        alpha=0.7,
        label="Исходная выборка",
        color="blue",
    )
    plt.hist(
        data2,
        bins=bins,
        edgecolor="black",
        alpha=0.5,
        label="Сгенерированная выборка",
        color="red",
    )

    plt.title("Гистограмма распределения частот для двух массивов")
    plt.xlabel("Значение")
    plt.ylabel("Частота")
    plt.grid(axis="y", alpha=0.75)  # Сетка для лучшего восприятия
    plt.legend()  # Легенда для различения массивов
    plt.show()


def densitys(data1, data2):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data1, color="blue", label="Исходная", fill=True, alpha=0.5)
    sns.kdeplot(data2, color="orange", label="Сгенерированная", fill=True, alpha=0.5)

    # Добавление элементов графика
    plt.title("График плотности вероятностей для двух выборок")
    plt.xlabel("Значение")
    plt.ylabel("Плотность вероятности")
    plt.legend()
    plt.grid()

    # Отображение графика
    plt.show()


def approximate_distribution(data):
    """
    Выполняет аппроксимацию закона распределения заданной случайной последовательности.

    :param data: Список чисел, представляющих случайную последовательность
    :return: Распределение и его параметры
    """
    # Вычисляем необходимые статистики
    mean = np.mean(data)
    variance = np.var(data)
    std_deviation = np.std(data)
    coefficient_of_variation = std_deviation / mean

    # Определяем распределение на основе коэффициента вариации
    if coefficient_of_variation == 0:
        print("Все значения в последовательности одинаковы, распределение невозможно.")
        return None

    if coefficient_of_variation < 0.5:
        # Используем равномерное распределение
        a = mean - np.sqrt(3 * variance)
        b = mean + np.sqrt(3 * variance)
        distribution = stats.uniform(loc=a, scale=b - a)
        print(f"Равномерное распределение: a={a}, b={b}")

    elif 0.5 <= coefficient_of_variation < 1:
        # Используем экспоненциальное распределение
        scale = mean  # Параметр масштаба
        distribution = stats.expon(scale=scale)
        print(f"Экспоненциальное распределение: scale={scale}")

    elif 1 <= coefficient_of_variation < 2:
        # Используем гипоэкспоненциальное распределение (2 параметра)
        k = 2  # Порядок
        lambda_param = 1 / mean  # Параметр
        distribution = stats.hypoexponential([lambda_param] * k)
        print(f"Гипоэкспоненциальное распределение: lambda={lambda_param}, k={k}")

    else:
        # Используем гиперэкспоненциальное распределение
        # Для гиперэкспоненциального распределения предположим 2 компонента
        p = 0.5  # Вероятность выбора первого распределения
        lambda1 = 1 / mean  # Параметр первого компонента
        lambda2 = lambda1 / coefficient_of_variation  # Параметр второго компонента
        distribution = stats.hypoexponential([lambda1, lambda2], [p, 1 - p])
        print(
            f"Гиперэкспоненциальное распределение: lambda1={lambda1}, lambda2={lambda2}, p={p}"
        )

    # Создаем гистограмму исходных данных
    plt.figure(figsize=(10, 6))
    plt.hist(
        data, bins=30, density=True, alpha=0.5, color="g", label="Гистограмма данных"
    )

    # Генерируем точки для графика плотности
    x = np.linspace(min(data), max(data), 1000)
    y = distribution.pdf(x)

    # Строим график плотности
    plt.plot(x, y, "r-", lw=2, label="Плотность распределения")

    plt.title("Сравнение гистограммы и плотности распределения")
    plt.xlabel("Значение")
    plt.ylabel("Плотность")
    plt.legend()
    plt.grid()
    plt.show()

    return distribution
