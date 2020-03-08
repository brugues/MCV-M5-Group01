import os
import glob
import numpy as np


def main(file):
    filename = os.path.join(file)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        easy_score = np.mean([float(number) for number in lines[0].rstrip().split(' ')])
        moderate_score = np.mean([float(number) for number in lines[1].rstrip().split(' ')])
        hard_score = np.mean([float(number) for number in lines[2].rstrip().split(' ')])
        return (easy_score + moderate_score + hard_score) / 3
    return np.zeros(3, dtype=np.float32)


if __name__ == '__main__':

    files = glob.glob('./cpp/output_7_classes_resnext101/*.txt')
    files.sort()
    maps = {}

    for file in files:
        maps[file] = main(file)

    for key, value in maps.items():
        print(key, ' ', value)
