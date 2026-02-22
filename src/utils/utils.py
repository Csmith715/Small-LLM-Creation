import math

def calc_sample_size(population_size: int,) -> float:
    z = 1.64485
    e = 0.1
    p = 0.5
    n0 = (z**2 * p * (1 - p)) / (e**2)
    _n = population_size
    n = (_n * n0) / (_n + n0 - 1)
    return math.ceil(n)
