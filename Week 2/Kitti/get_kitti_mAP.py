import os
import glob
import numpy as np


def main(file):
    filename = os.path.join(file)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            vLines = f.readlines()
            vLines = np.asarray([[float(number) for number in line.rstrip().split(' ')] for line in vLines])
        mean_scores = np.mean(vLines, axis=1)
        return mean_scores
    return np.zeros(3, dtype=np.float32)


if __name__ == '__main__':

    files = glob.glob('./cpp/output_all_classes_resnext101/plot/*.txt')
    files.sort()
    maps = {}

    for file in files:
        values = main(file)
        sum = 0.0
        for i in range(41):
            sum += values[i]
        maps[file] = sum/41

    for key, value in maps.items():
        print(key, ' ', value)
