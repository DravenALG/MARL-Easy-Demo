import argparse
import os
import numpy as np
from matplotlib import pyplot as plt

# parameters
parser = argparse.ArgumentParser()
parser.add_argument("--path", default="save/test/actor_loss.npy", type=str)
config = parser.parse_args()

if __name__ == '__main__':
    results = np.load(config.path)
    plt.subplot(1, 1, 1)
    plt.plot(results)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()
    plt.close()