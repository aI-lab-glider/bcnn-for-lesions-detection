from model.train import train
from model.train import ex


@ex.automain
def train_network():
    train()


if __name__ == '__main__':
    train_network()