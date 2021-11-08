import json

from bayesian_cnn_prometheus.bayesian_detector import BayesianDetector


# TODO ProxPxD find a proper name
def main():
    config = get_config()
    detector = BayesianDetector(config)
    detector.load_data()
    detector.create_model()
    detector.fit()


def get_config():
    with open('./config.json') as cf:
        config = json.load(cf)
    return config


if __name__ == '__main__':
    main()
