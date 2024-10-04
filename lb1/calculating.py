from collections import Counter
from scipy import stats
import math
import numpy as np


def Mathematical_expectation(data):
    # Вычисляем количество каждого уникального значения
    count = Counter(data)

    # Общая длина выборки
    total_count = len(data)

    # Вычисляем вероятности для каждого уникального значения
    probabilities = {value: freq / total_count for value, freq in count.items()}

    # Инициализируем математическое ожидание
    expectation = 0

    # Вычисляем математическое ожидание
    for value, prob in probabilities.items():
        expectation += value * prob

    return expectation


def Dispersion(data):
    # Вычисляем количество каждого уникального значения
    count = Counter(data)

    # Общая длина выборки
    total_count = len(data)

    # Вычисляем вероятности для каждого уникального значения
    probabilities = {value: freq / total_count for value, freq in count.items()}

    # Инициализируем математическое ожидание и математическое ожидание квадрата
    expectation = 0
    expectation_square = 0

    # Вычисляем математическое ожидание и математическое ожидание квадрата
    for value, prob in probabilities.items():
        expectation += value * prob
        expectation_square += (value**2) * prob

    # Вычисляем дисперсию
    dispersion = expectation_square - expectation**2

    return dispersion


def Standard_deviation(dispersion):
    # Вычисляем среднеквадратическое отклонение
    std_deviation = math.sqrt(dispersion)
    return std_deviation


def Coefficient_of_variation(std_deviation, expectation):
    # Проверка на нулевое математическое ожидание (чтобы избежать деления на ноль)
    if expectation == 0:
        raise ValueError(
            "Математическое ожидание равно нулю, коэффициент вариации не может быть вычислен."
        )

    # Вычисляем коэффициент вариации
    cv = (std_deviation / expectation) * 100
    return cv


def Confidence_interval(data, std_deviation, ME, confidence_level):
    n = len(data)  # Размер выборки

    # Критическое значение для нормального распределения
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # Вычисляем доверительный интервал
    margin_of_error = z_score * (std_deviation / math.sqrt(n))
    lower_bound = ME - margin_of_error
    upper_bound = ME + margin_of_error

    print(
        f"Доверительный интервал ({confidence_level*100}%): ({lower_bound}, {upper_bound})"
    )
    return (lower_bound, upper_bound)


def Relative_deviation(current_data):
    reference_data = np.random.normal(loc=100, scale=15, size=300)
    # Проверка, что размеры выборок одинаковы
    if len(current_data) != len(reference_data):
        raise ValueError("Размеры текущих и эталонных выборок должны совпадать.")

    # Рассчитываем относительные отклонения
    relative_deviations = []
    for current_value, reference_value in zip(current_data, reference_data):
        if reference_value == 0:  # Избегаем деления на ноль
            raise ValueError("Эталонное значение не должно быть равно нулю.")
        relative_deviation = (
            abs(current_value - reference_value) / abs(reference_value) * 100
        )
        relative_deviations.append(relative_deviation)

    for i in range(300):
        print(
            f"Текущее значение: {current_data[i]:.2f}, Эталонное значение: {reference_data[i]:.2f}, "
            f"Относительное отклонение: {relative_deviations[i]:.2f}%"
        )
    return relative_deviations
