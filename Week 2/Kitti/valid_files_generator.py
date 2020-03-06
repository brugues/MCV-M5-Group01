import os
import pathlib
import glob


def main():
    """
    Main
    """
    PATH = './'
    filenames = []

    for filepath in pathlib.Path(PATH).glob('**/*'):
        filenames.append(filepath.absolute())
    filenames.sort()
    print(filenames)

    file = open('./valid_file.txt', 'w')

    for filename in filenames:
        file.write(str(filename) + '\n')
    
    file.close()


if __name__ == '__main__':
    main()
