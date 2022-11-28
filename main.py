from test import run
from BinaryClassification.NormalDestribution.DataPrep import create_data_set
from BinaryClassification.NormalDestribution.Model import Model

if __name__ == '__main__':
    dataset = create_data_set((100, 100), 0.5)
    print(dataset)

    model = Model()
    model.fit(dataset=dataset, steps=30, lr=0.5, squeeze_rate=1.5)
    print(model.predict(x_array=dataset.testX, y_array=dataset.testY))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
