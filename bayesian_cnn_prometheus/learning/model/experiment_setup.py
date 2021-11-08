from sacred import Experiment

from bayesian_cnn_prometheus.constants import CONFIG_PATH

ex = Experiment()
ex.add_config(str(CONFIG_PATH.resolve()))
