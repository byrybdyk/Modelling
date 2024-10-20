import numpy as np
import scipy.stats as stats
import random
from scipy.stats import expon
from graph import *


def calculate_statistics(data2):
    n = len(data2)
    mean = np.mean(data2)
    variance = np.var(data2, ddof=1)
    std_dev = np.std(data2)
    coef_variation = (std_dev / mean) * 100
    conf_90 = (1 - 0.9) * (std_dev / np.sqrt(n))
    conf_95 = (1 - 0.95) * (std_dev / np.sqrt(n))
    conf_99 = (1 - 0.99) * (std_dev / np.sqrt(n))

    return {
        "mean": mean,
        "variance": variance,
        "std_dev": std_dev,
        "coef_variation": coef_variation,
        "conf_90": conf_90,
        "conf_95": conf_95,
        "conf_99": conf_99,
    }


def relative_deviation(estimate, reference):
    return (abs(estimate - reference) / reference) * 100


numbers = [
    170.024,
    79.539,
    130.762,
    90.089,
    163.334,
    136.418,
    200.392,
    293.442,
    219.744,
    380.663,
    195.666,
    322.105,
    490.399,
    364.577,
    58.487,
    72.014,
    177.088,
    183.206,
    22.328,
    98.121,
    140.249,
    221.596,
    86.552,
    90.440,
    233.314,
    16.709,
    169.271,
    47.460,
    275.898,
    40.941,
    313.818,
    60.952,
    34.950,
    54.336,
    329.629,
    192.693,
    192.847,
    194.905,
    110.372,
    209.361,
    180.888,
    288.685,
    130.890,
    584.312,
    380.720,
    165.779,
    236.191,
    94.679,
    111.330,
    248.540,
    361.398,
    98.416,
    63.981,
    303.634,
    86.800,
    208.942,
    263.589,
    88.757,
    43.298,
    131.125,
    214.791,
    208.163,
    223.119,
    179.650,
    40.163,
    189.742,
    243.522,
    112.035,
    340.319,
    404.583,
    294.998,
    72.711,
    251.290,
    20.030,
    146.130,
    100.434,
    150.415,
    199.821,
    87.417,
    257.168,
    15.892,
    14.836,
    51.800,
    418.550,
    7.929,
    62.499,
    174.340,
    18.645,
    159.722,
    46.789,
    96.408,
    116.752,
    96.128,
    164.833,
    68.306,
    420.383,
    118.449,
    42.489,
    222.866,
    141.328,
    115.424,
    58.955,
    38.136,
    126.614,
    103.409,
    116.393,
    74.402,
    22.548,
    92.365,
    82.353,
    126.877,
    39.683,
    179.549,
    170.339,
    92.796,
    292.580,
    35.073,
    35.910,
    365.053,
    99.318,
    170.044,
    439.059,
    470.488,
    357.771,
    430.030,
    73.933,
    83.735,
    52.641,
    93.914,
    38.858,
    80.181,
    49.148,
    323.159,
    195.029,
    129.108,
    98.240,
    441.188,
    153.093,
    199.637,
    79.140,
    120.930,
    83.837,
    89.329,
    239.461,
    160.427,
    262.283,
    155.024,
    11.625,
    241.299,
    22.612,
    58.050,
    39.053,
    243.639,
    218.856,
    76.824,
    327.907,
    136.962,
    371.455,
    101.572,
    329.667,
    115.963,
    161.151,
    296.856,
    119.736,
    109.337,
    117.836,
    176.364,
    117.964,
    190.672,
    62.132,
    171.453,
    330.865,
    405.409,
    223.720,
    239.713,
    199.497,
    49.795,
    61.673,
    52.205,
    72.039,
    99.700,
    233.580,
    150.956,
    198.003,
    201.332,
    108.135,
    80.208,
    173.854,
    144.678,
    175.169,
    129.944,
    142.436,
    79.915,
    86.711,
    56.352,
    43.214,
    140.616,
    206.702,
    168.478,
    137.919,
    51.298,
    244.433,
    207.864,
    97.953,
    32.952,
    255.189,
    133.041,
    100.125,
    140.072,
    196.401,
    119.388,
    175.919,
    266.864,
    199.475,
    101.431,
    365.202,
    270.031,
    177.405,
    175.620,
    140.164,
    289.649,
    139.261,
    205.318,
    132.671,
    253.934,
    172.386,
    119.386,
    90.815,
    94.161,
    301.453,
    40.729,
    237.380,
    52.348,
    13.091,
    270.564,
    637.907,
    99.353,
    96.998,
    214.060,
    215.158,
    166.624,
    128.606,
    173.574,
    143.082,
    478.023,
    322.467,
    284.337,
    46.326,
    141.568,
    202.882,
    18.524,
    49.681,
    138.104,
    334.007,
    43.652,
    650.381,
    116.376,
    53.596,
    436.236,
    158.582,
    192.722,
    205.115,
    70.249,
    138.277,
    42.582,
    264.834,
    151.359,
    125.369,
    266.426,
    364.757,
    61.749,
    10.127,
    65.263,
    40.274,
    55.823,
    153.138,
    218.336,
    220.224,
    376.028,
    89.408,
    97.385,
    112.528,
    25.728,
    109.802,
    413.923,
    204.841,
    142.826,
    265.614,
    48.349,
    348.667,
    257.964,
    186.919,
    132.503,
    486.858,
    219.577,
    123.580,
    151.368,
    109.114,
    288.021,
    138.902,
]


# def generation_data():

#     a1 = 150
#     a2 = 40
#     q1 = 0.04
#     q2 = 0.96
#     generator2 = expon.rvs(scale=a1, loc=0, size=300)
#     generator1 = expon.rvs(scale=a2, loc=0, size=300)
#     result = []
#     for i in range(1, 301):
#         number = random.random()
#         idx = random.randint(0, 299)
#         if number <= q1:
#             result.append(generator1[idx])
#         else:
#             result.append(generator2[idx])
#     # for i in result:
#     #     print(i)
#     return result


def generate_erlang_sequence(size=300):
    """
        Генерирует случайную последовательность из заданного
    количества значений в соответствии с распределением Эрланга k-го
    порядка.
        :param k: Коэффициент формы (целое число)
        :param scale: Масштабный параметр (темп для лямбда)
        :param size: Количество генерируемых случайных значений
        :return: Массив случайных чисел в соответствии с
    распределением Эрланга
    """
    # Параметры gamma распределения: shape = k, scale = 1/lambda
    return np.random.gamma(shape=3, scale=56.52, size=size)


sorted_numbers = sorted(numbers)


# Подвыборки
samples = {
    10: numbers[:10],
    20: numbers[:20],
    50: numbers[:50],
    100: numbers[:100],
    200: numbers[:200],
    300: numbers[:300],
}



# Эталонная выборка для 300 элементов
reference_stats = calculate_statistics(samples[300])

# Результаты
for n, sample in samples.items():
    stats_sample = calculate_statistics(sample)
    rel_dev_mean = relative_deviation(stats_sample["mean"], reference_stats["mean"])

    print(f"\nСтатистика для {n} элементов:")
    print(f"Математическое ожидание: {stats_sample['mean']:.3f}")
    print(f"Дисперсия: {stats_sample['variance']:.3f}")
    print(f"Среднеквадратическое отклонение: {stats_sample['std_dev']:.3f}")
    print(f"Коэффициент вариации: {stats_sample['coef_variation']:.3f}%")
    print(f"Доверительный интервал 90%: ({stats_sample['conf_90']:.3f})")
    print(f"Доверительный интервал 95%: ({stats_sample['conf_95']:.3f})")
    print(f"Доверительный интервал 99%: ({stats_sample['conf_99']:.3f})")
    print(f"Относительное отклонение математического ожидания: {rel_dev_mean:.3f}%")

new_data = generate_erlang_sequence()

analyze_sequence(numbers)
analyze_sequence(new_data)

plot_frequency_histogram(numbers)
plot_frequency_histogram(new_data)

samples_generated = {
    10: new_data[:10],
    20: new_data[:20],
    50: new_data[:50],
    100: new_data[:100],
    200: new_data[:200],
    300: new_data[:300],
}

generated_reference_stats = calculate_statistics(new_data)

print(f"\nСтатистика для сгенерерованных данных\n{"-"*40}")
for n, sample in samples_generated.items():
    stats_sample = calculate_statistics(sample)
    rel_dev_mean = relative_deviation(stats_sample["mean"], generated_reference_stats["mean"])

    print(f"\nСтатистика для {n} элементов:")
    print(f"Математическое ожидание: {stats_sample['mean']:.3f}")
    print(f"Дисперсия: {stats_sample['variance']:.3f}")
    print(f"Среднеквадратическое отклонение: {stats_sample['std_dev']:.3f}")
    print(f"Коэффициент вариации: {stats_sample['coef_variation']:.3f}%")
    print(f"Доверительный интервал 90%: ({stats_sample['conf_90']:.3f})")
    print(f"Доверительный интервал 95%: ({stats_sample['conf_95']:.3f})")
    print(f"Доверительный интервал 99%: ({stats_sample['conf_99']:.3f})")
    print(f"Относительное отклонение математического ожидания: {rel_dev_mean:.3f}%")

autocorrelation_analysis(new_data)

analyze_sequences(numbers, new_data)
plot_frequency_histograms(numbers, new_data)
densitys(numbers, new_data)
