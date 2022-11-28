import tensorflow as tf
import strawberryfields as sf
import numpy as np
from strawberryfields.ops import Sgate, BSgate, Dgate, Pgate

from qmlt.tf.helpers import make_param
from qmlt.tf import CircuitLearner

from BinaryClassification.NormalDestribution.DataPrep import DataSet


class Model:

    def __init__(self, params=None):
        self.params, self.learner, self.squeeze_rate = None, None, None
        self.lr, self.steps, self.learner = None, None, None
        if params is not None:
            self.params = params

    def _circuit(self, X, params):
        eng, q = sf.Engine(2)

        with eng:
            Dgate(X[:, 0], 0.) | q[0]
            Dgate(X[:, 1], 0.) | q[1]
            BSgate(phi=params[0]) | (q[0], q[1])
            BSgate() | (q[0], q[1])

            Sgate(self.squeeze_rate, X[:, 0]) | q[0]
            Sgate(self.squeeze_rate, X[:, 1]) | q[1]
            BSgate(params[0], params[7]) | (q[0], q[1])
            Dgate(params[1]) | q[0]
            Dgate(params[2]) | q[1]
            Pgate(params[3]) | q[0]
            Pgate(params[4]) | q[1]
            # Kgate(params[5]) | q[0]
            # Kgate(params[6]) | q[1]

        num_inputs = X.get_shape().as_list()[0]
        state = eng.run('tf', cutoff_dim=10, eval=False, batch_size=num_inputs)

        p0 = state.fock_prob([0, 2])
        p1 = state.fock_prob([2, 0])
        normalization = p0 + p1 + 1e-10
        circuit_output = p1 / normalization

        return circuit_output

    def _myloss(self, circuit_output, targets):
        return tf.losses.mean_squared_error(labels=circuit_output, predictions=targets)

    def _outputs_to_predictions(self, outpt):
        return tf.round(outpt)

    def score_model(self, dataset: DataSet):
        test_score = self.learner.score_circuit(X=dataset.testX, Y=dataset.testY,
                                           outputs_to_predictions=self._outputs_to_predictions)
        # The score_circuit() function returns a dictionary of different metrics.
        print("\nPossible scores to print: {}".format(list(test_score.keys())))
        # We select the accuracy and loss.
        print("Accuracy on test set: ", test_score['accuracy'])
        print("Loss on test set: ", test_score['loss'])

    def predict(self, x_array: np.array, y_array: np.array):
        prediction = self.learner.score_circuit(X=x_array, Y=y_array,
                                           outputs_to_predictions=self._outputs_to_predictions)
        return prediction['predictions']

    def _create_hyperparams(self, lr: float, decay=None):
        if decay is not None:
            hyperparams = {'circuit': self._circuit,
                           'task': 'supervised',
                           'loss': self._myloss,
                           'optimizer': 'SGD',
                           'init_learning_rate': lr,
                           'decay': decay,
                           'print_log': True,
                           'log_every': 1,
                           'warm_start': False
                           }
        else:
            hyperparams = {'circuit': self._circuit,
                           'task': 'supervised',
                           'loss': self._myloss,
                           'optimizer': 'SGD',
                           'init_learning_rate': lr,
                           'print_log': True,
                           'log_every': 1,
                           'warm_start': False
                           }
        return hyperparams

    def fit(self, dataset: DataSet, steps: int, lr: float, batch_size=None):

        self.params = [make_param(name='phi'+str(i), constant=.7) for i in range(9)]
        hyperparams = self._create_hyperparams(lr=lr)
        self.learner = CircuitLearner(hyperparams=hyperparams)
        if batch_size is not None:
            self.learner.train_circuit(X=dataset.trainX, Y=dataset.trainY,
                                       steps=steps, batch_size=batch_size)
        else:
            self.learner.train_circuit(X=dataset.trainX, Y=dataset.trainY,
                                       steps=steps, batch_size=len(dataset.trainY))
