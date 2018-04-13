import csv
import random as rdm
import numpy as np
from HistogramApprox import HistogramApprox
from MultidimGaussian import MultidimGaussian
import matplotlib.pyplot as plt


class Classifier(object):

    input_file = None
    data = None
    train_percent = 0.8
    train_idx = []
    test_idx = []
    priori_pulsar = 0
    priori_nonpulsar = 0
    confussion_matrix = None
    histoApprox = None
    muldimGauss = None

    def __init__(self, input_file):
        self.input_file = input_file
        self.set_info()
        self.cost_ratio = np.random.uniform(low=0.0, high=1.0, size=50)
        self.histo_fpr = []
        self.histo_tpr = []
        self.gauss_fpr = []
        self.gauss_tpr = []

    def set_info(self):

        with open(self.input_file, 'rt') as f:
            self.data = list(csv.reader(f))
        rdm.shuffle(self.data)
        n_data = len(self.data)
        self.train_idx = np.arange(0, int(n_data*.8))
        self.test_idx = np.arange(int(n_data*.8), n_data)
        self.set_piors()

    def set_piors(self):
        n_pulsars = len([self.data[i] for i in self.train_idx if int(self.data[i][8])==1])
        n_nonpulsars = len([self.data[i] for i in self.train_idx if int(self.data[i][8]) == 0])
        train_len = len(self.train_idx)
        self.priori_pulsar = float(n_pulsars)/float(train_len)
        self.priori_nonpulsar = float(n_nonpulsars) / float(train_len)

    def bayessianEvaluation(self, theta):
        histoApprox = HistogramApprox()
        histoApprox.set_pdfs(self.data, self.train_idx)
        return histoApprox.evaluation(self.test_idx, self.data, self.priori_pulsar, self.priori_nonpulsar, theta)

    def multiGaussianEvaluation(self, theta):

        multidimGauss = MultidimGaussian(self.priori_pulsar, self.priori_nonpulsar)
        multidimGauss.calc_stats(self.data, self.train_idx)
        return multidimGauss.evaluation(self.data, self.test_idx,  theta)

    def evaluation(self, theta):

        [tpr, fpr]=self.bayessianEvaluation(theta)
        self.histo_tpr.append(tpr)
        self.histo_fpr.append(fpr)

        [tpr, fpr] =self.multiGaussianEvaluation(theta)
        self.gauss_tpr.append(tpr)
        self.gauss_fpr.append(fpr)

    def roc_curves(self):

        thetas = self.cost_ratio*(self.priori_nonpulsar/self.priori_pulsar)
        for t in thetas:
            self.evaluation(t)

        plt.scatter(self.histo_fpr, self.histo_tpr,  c="b", alpha=0.5, marker='o')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Bayessian Approach")
        plt.savefig('bayes.png')

        plt.clf()
        plt.scatter(self.gauss_fpr, self.gauss_tpr,  c="r", alpha=0.5, marker='o')
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Multidimensional Gaussian")
        plt.savefig('gauss.png')
