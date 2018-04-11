import numpy as np


class MultidimGaussian(object):


    def __init__(self):
        self.mus_pulsar= np.zeros(8)
        self.mus_nonpulsar = np.zeros(8)

        self.sigmas_pulsar = np.zeros(8)
        self.sigmas_nonpulsar = np.zeros(8)
        self.inv = None
        self.inv_non = None
        self.fpr = 0 #falsos positivos
        self.vpr = 0 #verdaderos positivos
        self.negatives = 0
        self.positives = 0

    def calc_stats(self, data, train_idx):

        values_pulsar = []
        values_nonpulsar = []

        for i in train_idx:
            if data[i][8] == '1':
                values_pulsar.append(np.array(data[i][:8]).astype(np.float))
            else:
                values_nonpulsar.append(np.array(data[i][:8]).astype(np.float))

        values_pulsar = np.array(values_pulsar)

        self.mus_pulsar = np.mean(values_pulsar, axis=0)
        self.mus_nonpulsar = np.mean(values_nonpulsar, axis=0)
        self.sigmas_pulsar = np.cov(values_pulsar, rowvar=0)
        self.sigmas_nonpulsar = np.cov(values_nonpulsar, rowvar=0)

        self.inv_non = np.linalg.inv(self.sigmas_nonpulsar)
        self.inv = np.linalg.inv(self.sigmas_pulsar)


    def calc_prob_nonpulsar(self,x):

        diff = x - self.mus_nonpulsar
        exp = np.exp(-.5*np.dot(diff.T,np.dot(self.inv_non, diff)))
        fact = np.power((2*np.pi), 4)*(np.linalg.det(self.sigmas_nonpulsar)**.5)
        return exp/fact
        #print(pow)

    def calc_prob_pulsar(self,x):

        diff = x - self.mus_pulsar
        exp = np.exp(-.5*np.dot(diff.T,np.dot(self.inv, diff)))
        fact = np.power((2*np.pi), 4)*(np.linalg.det(self.sigmas_pulsar)**.5)
        return exp/fact

    def evaluation(self, data, test_idx, theta):

        for i in test_idx:
            d = data[i]
            post_non= self.calc_prob_nonpulsar(np.array(d[:8]).astype('float64'))
            if post_non==0.0:
                post_non = 0.000001

            post= self.calc_prob_pulsar(np.array(d[:8]).astype('float64'))
            post_ratio = post/post_non
            post_ratio>theta
            print(d[8] == '1')






