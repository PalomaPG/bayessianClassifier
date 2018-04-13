import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class HistogramApprox(object):

    pdfs_nonpulsars = []
    pdfs_pulsars = []

    def __init__(self):
        self.fpr = 0 #falsos positivos
        self.tpr = 0 #verdaderos positivos
        self.true_negatives = 0
        self.false_negatives = 0

    def histo_prob(self, att, show, pulsar, data, train_idx):

        hist, bin_edges = np.histogram(np.array(
            [data[i][att] for i in train_idx if float(data[i][8]) == pulsar]
            ).astype(np.float), bins='auto')
        [mu, sigma] = norm.fit(hist)
        if show:
            [n, bins, patches] = plt.hist(hist, len(bin_edges), density=True)
            # add a 'best fit' line
            y = norm.pdf(bins, mu, sigma)
            plt.plot(bins, y, 'r--', linewidth=2)
            plt.grid(True)
            plt.show()

        return [mu, sigma]

    def set_pdfs(self, data, train_idx):

        for i in np.arange(0,8):
            [mu, sigma] = self.histo_prob(i, False, 1, data, train_idx)
            self.pdfs_pulsars.append(norm(mu, sigma))
            [mu, sigma] = self.histo_prob(i, False, 0, data, train_idx)
            self.pdfs_nonpulsars.append(norm(mu, sigma))

    def is_pulsar(self, d, priori_pulsar, priori_nonpulsar, theta):

        post_pulsar = priori_pulsar
        post_nonpulsar = priori_nonpulsar
        for i in np.arange(0, 8):
            post_pulsar = self.pdfs_pulsars[i].pdf(float(d[i]))*post_pulsar
            post_nonpulsar = self.pdfs_nonpulsars[i].pdf(float(d[i]))*post_nonpulsar

        post_ratio = post_pulsar/post_nonpulsar

        return post_ratio > theta

    def evaluation(self, test_idx, data, priori_pulsar, priori_nonpulsar, theta):

        for i in test_idx:
            d = data[i]
            is_pulsar = self.is_pulsar(d, priori_pulsar, priori_nonpulsar, theta)
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
