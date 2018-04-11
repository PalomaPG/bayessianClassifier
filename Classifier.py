import csv
import random as rdm
import numpy as np
from HistogramApprox import HistogramApprox
from MultidimGaussian import MultidimGaussian


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
        self.cost_ratio = np.random.uniform(low=0.01, high=100, size=20)
        print(self.cost_ratio)

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
        histoApprox.evaluation(self.test_idx, self.data, self.priori_pulsar, self.priori_nonpulsar, theta)

    def multiGaussianEvaluation(self, theta):

        multidimGauss = MultidimGaussian()
        multidimGauss.calc_stats(self.data, self.train_idx)
        multidimGauss.evaluation(self.data, self.test_idx,  theta)


    def evaluation(self, theta):

        #theta = self.priori_nonpulsar/self.priori_pulsar
        self.bayessianEvaluation(theta)
        self.multiGaussianEvaluation(theta)

    def roc_curves(self):

        thetas = self.cost_ratio*(self.priori_nonpulsar/self.priori_pulsar)

    '''
    def check_prob(self, mu, sigma, x):
        normie = norm(mu, sigma)
        return normie.pdf(x)
    '''

