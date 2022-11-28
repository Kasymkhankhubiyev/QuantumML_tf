"""
Here I prepare Data for work flow
"""
import numpy as np
from typing import NamedTuple


class DataSet(NamedTuple):
    trainX: np.array
    trainY: np.array
    testX: np.array
    testY: np.array


def _create_normal_distributed_data(classes_scale: tuple, intersect_rate: float) -> DataSet:
    np.random.seed(0)
    l1, l2 = classes_scale[0], classes_scale[1]
    n = 2
    drop = intersect_rate

    X1 = np.array([[-1, -1]]) + drop * np.random.randn(l1, n)
    X2 = np.array([[1, 1]]) + drop * np.random.randn(l2, n)

    # конкатенируем все в одну матрицу
    # при этом по 20 точек оставим на тест/валидацию
    X = np.vstack((X1[10:], X2[10:]))
    ValX = np.vstack((X1[:10], X2[:10]))

    # конкатенируем все в один столбец с соответствующими значениями для класса 0 или 1
    y = np.hstack([[0] * (l1 - 10), [1] * (l2 - 10)])
    ValY = np.hstack([[0] * 10, [1] * 10])

    return DataSet(trainX=X, trainY=y, testX=ValX, testY=ValY)


def create_data_set(classes_scale: tuple, intersect_rate: float) -> DataSet:
    # из этого можно сделать некоторую общую библиотеку
    return _create_normal_distributed_data(classes_scale, intersect_rate)