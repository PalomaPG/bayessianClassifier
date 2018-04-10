import sys
from BayesianClassifier import BayesianClassifier


def main(input_file):

    bc = BayesianClassifier(input_file)
    bc.evaluation()


if __name__=="__main__":

    main(sys.argv[1])