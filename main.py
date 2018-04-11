import sys
from Classifier import Classifier


def main(input_file):

    c = Classifier(input_file)
    c.evaluation()


if __name__=="__main__":

    main(sys.argv[1])