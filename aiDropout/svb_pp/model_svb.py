import numpy as np
from scipy.special import digamma
import time

def dirichlet_expectation(alpha):
    if (len(alpha.shape) == 1):
        return digamma(alpha) - digamma(sum(alpha))
    return (digamma(alpha) - digamma(np.sum(alpha, axis=1))[:, np.newaxis])

class Model:

    def __init__(self, num_topic, n_term, batch_size, n_infer, alpha, eta):
        self.num_topic = num_topic
        self.n_term = n_term
        self.batch_size = batch_size
        self.n_infer = n_infer
        self.alpha = alpha

        self.eta = eta
        self.lamda = np.random.gamma(100., 1./100., (self.num_topic, self.n_term))

    def do_e_step(self, wordinds, wordcnts):

        sum_phi_i = np.zeros((self.num_topic, self.n_term)) # kj-element is sum(I[w_dn=j]*phi_dnk, d=1...D, n=1...Nd)

        gamma = np.random.gamma(100., 1./100., (self.batch_size, self.num_topic))
        expElogtheta = np.exp(dirichlet_expectation(gamma))
        expElogbeta = np.exp(dirichlet_expectation(self.lamda))

        print ('Updating local variable...')
        t1 = time.time()
        for d in range(self.batch_size):
            inds = wordinds[d]
            cnts = wordcnts[d]

            gamma_d = gamma[d, :]
            expElogtheta_d = expElogtheta[d, :]
            expElogbeta_d = expElogbeta[:, inds]

            for it in range(self.n_infer):
                phi_d = expElogtheta_d * expElogbeta_d.transpose() + 1e-10
                phi_d /= np.sum(phi_d, axis=1)[:, np.newaxis]
                gamma_d = self.alpha + np.dot(cnts, phi_d)
                expElogtheta_d = np.exp(dirichlet_expectation(gamma_d))

            sum_phi_i[:, inds] += phi_d.transpose() * cnts

        t2 = time.time()
        print ('Total time update: ', t2-t1)
        return sum_phi_i
        
    def update_stream(self, wordinds, wordcnts, minibatch):
        sum_phi_i = self.do_e_step(wordinds, wordcnts)
        if (minibatch == 1):
            self.lamda = self.eta + sum_phi_i
        else:
            self.lamda = self.lamda + sum_phi_i