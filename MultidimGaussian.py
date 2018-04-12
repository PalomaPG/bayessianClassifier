import numpy as np
from functools import reduce

class MultidimGaussian(object):

    def __init__(self,  prior, prior_non):
        self.mus_pulsar= np.zeros(8)
        self.mus_nonpulsar = np.zeros(8)

        self.sigmas_pulsar = np.zeros(8)
        self.sigmas_nonpulsar = np.zeros(8)
        self.inv = None
        self.inv_non = None
        self.fpr = 0 #falsos positivos
        self.tpr = 0 #verdaderos positivos
        self.true_negatives = 0
        self.false_negatives = 0
        self.prior_non = prior_non
        self.prior = prior

    def calc_stats(self, data, train_idx):

        values_pulsar = []
        values_nonpulsar = []

        for i in train_idx:
            if data[i][8] == '1':
                values_pulsar.append(np.array(data[i][:8]).astype('float64'))
            else:
                values_nonpulsar.append(np.array(data[i][:8]).astype('float64'))

        values_pulsar = np.array(values_pulsar, dtype='float64')
        values_nonpulsar = np.array(values_nonpulsar, dtype='float64')

        self.mus_pulsar = np.mean(values_pulsar, axis=0)
        self.mus_nonpulsar = np.mean(values_nonpulsar, axis=0)
        self.sigmas_pulsar = np.cov(values_pulsar.T)
        self.sigmas_nonpulsar = np.cov(values_nonpulsar.T)
        self.inv_non = np.linalg.inv(self.sigmas_nonpulsar)
        self.inv = np.linalg.inv(self.sigmas_pulsar)

    def ratio_prob(self,  x):
        diff = x - self.mus_nonpulsar
        non_exp = np.exp(-0.5 * np.dot(diff.T, np.dot(self.inv_non, diff)), dtype='float64')
        non_fact = (np.linalg.det(self.sigmas_nonpulsar) ** .5)
        diff = x - self.mus_pulsar
        exp = np.exp(-0.5 * np.dot(diff.T, np.dot(self.inv, diff)), dtype='float64')
        fact = (np.linalg.det(self.sigmas_pulsar) ** .5)


        return (exp/non_exp)*(non_fact/fact)*(self.prior/self.prior_non)

    def evaluation(self, data, test_idx, theta):

        for i in test_idx:
            d = np.array(data[i]).astype('float64')
            post_ratio = self.ratio_prob(d[:8])
            p_nonpulsar = 1/(1+post_ratio)
            p_pulsar = 1-p_nonpulsar
            is_pulsar = p_pulsar/p_nonpulsar > theta

            if not is_pulsar:
                if int(d[8]) == 1:
                    self.false_negatives = self.false_negatives + 1
                else:
                    self.true_negatives = self.true_negatives + 1

            if is_pulsar:
                if int(d[8]) == 1:
                    self.tpr = self.tpr + 1
                else:
                    self.fpr = self.fpr + 1

        self.tpr = float(self.tpr)/float(self.false_negatives + self.tpr)
        self.fpr = float(self.fpr)/float(self.true_negatives + self.fpr)

        return [self.tpr, self.fpr]






