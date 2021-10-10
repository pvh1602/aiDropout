import numpy as np
from scipy.special import digamma


def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return digamma(alpha) - digamma(sum(alpha))
    return (digamma(alpha) - digamma(np.sum(alpha, axis=1))[:, np.newaxis])


class Stream:

    def __init__(self, alpha, lamda, num_topic, n_term, n_infer):
        self.alpha = alpha
        self.lamda = lamda
        self.num_topic = num_topic
        self.n_infer = n_infer
        self.n_term = n_term
        self.beta = lamda / np.sum(lamda, axis = 1)[:, np.newaxis]

    def do_e_step(self, batchsize, wordinds, wordcnts):

        gamma = np.random.gamma(100., 1./100., (batchsize, self.num_topic))
        theta = np.zeros((batchsize, self.num_topic))
        expElogtheta = np.exp(dirichlet_expectation(gamma))
        expElogbeta = np.exp(dirichlet_expectation(self.lamda))

        for d in range(batchsize):
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

            theta[d] = gamma_d / sum(gamma_d) # normalization obtain theta

        return theta

    def compute_doc(self, theta_d, wordinds, wordcnts):
        """
        Compute log predictive probability for each document in 'w_ho' part.
        """
        ld2 = 0
        frequency = np.sum(wordcnts)
        for i in range(len(wordinds)):
            p = np.dot(theta_d, self.beta[:, wordinds[i]])
            ld2 += wordcnts[i]*np.log(p)
        if (frequency == 0):
            return ld2
        else:
            return ld2/frequency

    def compute_perplexity(self, wordinds1, wordcnts1, wordinds2, wordcnts2):
        """
        Compute log predictive probability for all documents in 'w_ho' part.
        """
        batch_size = len(wordinds1)
        # e step
        theta = self.do_e_step(batch_size, wordinds1, wordcnts1) + 1e-10
        # compute perplexity
        LD2 = 0
        for d in range(batch_size):
            LD2 += self.compute_doc(theta[d], wordinds2[d], wordcnts2[d])
        return LD2/batch_size
