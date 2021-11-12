from sacred import Experiment

from bayesian_cnn_prometheus.constants import Paths

ex = Experiment()
ex.add_config(str(Paths.CONFIG_PATH.resolve()))
