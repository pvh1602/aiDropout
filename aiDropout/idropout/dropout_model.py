import numpy as np
from numpy.random import seed
from scipy.special import digamma
import time
from numpy import linalg as LA

seed(1)

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return digamma(alpha) - digamma(sum(alpha))
    return (digamma(alpha) - digamma(np.sum(alpha, axis=1))[:, np.newaxis])

def softmax_convert(matrix):
    matrix_max = np.max(matrix, axis = 1)[:, np.newaxis]
    temp = np.exp(matrix - matrix_max)
    matrix_softmax = temp / np.sum(temp, axis=1)[:, np.newaxis]
    return matrix_softmax

class Model:

    def __init__(self, num_topic, n_term, batch_size, n_infer, learning_rate, alpha, sigma, rate, type_model):
        
        self.num_topic = num_topic
        self.n_term = n_term
        self.batch_size = batch_size

        self.n_infer = n_infer
        self.learning_rate = learning_rate

        self.alpha = alpha
        self.sigma = sigma

        self.type = type_model

        self.rate = rate
        
        self.beta = softmax_convert(np.random.rand(self.num_topic, self.n_term))
        self.beta_drop = softmax_convert(self.beta)


    def do_e_step(self, wordinds, wordcnts):

        gamma = np.random.gamma(100., 1./100., (self.batch_size, self.num_topic))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        beta_t = np.copy(self.beta)

        # print ('rate = ', self.rate)
        
        p = self.rate

        matrix_drop = np.random.binomial(1, p ,size=(self.num_topic, self.n_term))
        matrix_positive = np.where(matrix_drop == 1)
        matrix_negative = np.where(matrix_drop == 0)
        
        if (self.type == 0 or self.type == 2):
            matrix_drop[matrix_positive[0], matrix_positive[1]] = 1
        if (self.type == 1):
            matrix_drop[matrix_positive[0], matrix_positive[1]] = 1.0 / p
        if (self.type == 2):
            beta_t[matrix_negative[0], matrix_negative[1]] = 0

        pi_t = matrix_drop
        start = time.time()
        for i in range(10):
            if (self.type == 0 or self.type == 2):
                beta_drop_t = softmax_convert(self.rate * beta_t)
            if (self.type == 1):
                beta_drop_t = softmax_convert(beta_t)

            print ('Updating local variable ...')      
            for d in range(self.batch_size):
                inds = wordinds[d]
                cnts = wordcnts[d]
                gamma_d = gamma[d, :]
                Elogtheta_d = Elogtheta[d, :]
                expElogtheta_d = expElogtheta[d, :]
                # gamma_d = np.ones(self.num_topic) * self.alpha + float(np.sum(cnts)) / self.num_topic
                # expElogtheta_d = np.exp(dirichlet_expectation(gamma_d))
                beta_d = beta_drop_t[:, inds]

                for i in range(self.n_infer):
                    phi_d = expElogtheta_d * beta_d.transpose() + 1e-10
                    phi_d /= np.sum(phi_d, axis=1)[:, np.newaxis]
                    gamma_d = self.alpha + np.dot(cnts, phi_d)
                    expElogtheta_d = np.exp(dirichlet_expectation(gamma_d))
                gamma[d] = gamma_d
                expElogtheta[d] = expElogtheta_d

            print ('Updating parameter...')
            beta_t = self.gradient_ascent(pi_t, beta_t, beta_drop_t, wordinds, wordcnts, gamma, expElogtheta)

        end = time.time()
        print ('Total time update: %f' % (end - start))

        return beta_t

    def gradient_ascent(self, pi_t, beta_t, beta_drop_t, wordinds, wordcnts, gamma, expElogtheta):


        d = self.num_topic * self.n_term

        # USING ADAGRAD OPTIMIZATION
        sum_squares_grad_beta = (1e-8)*np.ones_like(beta_t)
        for j in range(self.n_infer):
            grad_beta_t = self.compute_gradient_beta(pi_t, beta_t, beta_drop_t, wordinds, wordcnts, gamma, expElogtheta)
            sum_squares_grad_beta += grad_beta_t**2
            lr_adapted = self.learning_rate / (np.sqrt(sum_squares_grad_beta))
            beta_t += lr_adapted*grad_beta_t
            pi_t = np.random.binomial(1, self.rate, size=(self.num_topic, self.n_term))
        # print ('[---] Update gradient beta done!')

        # # USING ADAM
        # beta_1 = 0.9
        # beta_2 = 0.999 
        # m_beta = np.zeros_like(beta_t)
        # v_beta = np.zeros_like(beta_t)
        # for t in range(100):
        #     t += 1
        #     grad_beta_t = self.compute_gradient_beta(pi_t, beta_t, beta_drop_t, wordinds, wordcnts, gamma, expElogtheta)
        #     m_beta = beta_1*m_beta - (1 - beta_1)*grad_beta_t
        #     v_beta = beta_2*v_beta + (1 - beta_2)*grad_beta_t**2
        #     m_beta_hat = m_beta / (1 - np.power(beta_1, t))
        #     v_beta_hat = v_beta / (1 - np.power(beta_2, t))

        #     beta_t -= 0.001*m_beta_hat / (np.sqrt(v_beta_hat) + 1e-8)
        # # print ('[---] Update gradient beta done!')


        return beta_t

    def compute_gradient_beta(self, pi_t, beta_t, beta_drop_t, wordinds, wordcnts, gamma, expElogtheta):

        sum_phi = np.zeros((self.num_topic, 1)) # k-element is sum(phi_dnk, d=1...D, n=1...Nd)
        sum_phi_i = np.zeros((self.num_topic, self.n_term)) # kj-element is sum(I[w_dn=j]*phi_dnk, d=1...D, n=1...Nd)

        for d in range(self.batch_size):
            inds = wordinds[d]
            cnts = wordcnts[d]
            expElogtheta_d = expElogtheta[d]
            beta_d = beta_drop_t[:, inds]

            phi_d = expElogtheta_d * beta_d.transpose() + 1e-10 # infer phi_d one more time
            phi_d /= np.sum(phi_d, axis=1)[:, np.newaxis] # shape len(ids)xK

            # for n, term in enumerate(inds):
            #     temp = cnts[n] * phi_d[n]
            #     sum_phi += temp[:, np.newaxis]
            #     # sum_phi_i[:, term] += cnts[n] * phi_d[n]

            temp = cnts * phi_d.transpose()
            sum_phi += np.sum(temp, axis = 1)[:, np.newaxis]
            sum_phi_i[:, inds] += phi_d.transpose() * cnts

        mul = beta_t*pi_t
        mul_max = np.max(mul, axis=1)[:, np.newaxis]
        mul_norm = mul - mul_max
        
        sum_exp = np.sum(np.exp(mul_norm), axis=1)[:, np.newaxis] + 1e-10

        grad_beta_t = -(beta_t - self.beta)/(self.sigma) + pi_t*(sum_phi_i - sum_phi*np.exp(mul_norm)/sum_exp)

        return grad_beta_t / self.batch_size


    def update_stream(self, wordinds, wordcnts, minibatch, rate):
        # iif (minibatch <= 5):
        #     self.rate = 1
        # else:
        #     self.rate = rate
        beta_t = self.do_e_step(wordinds, wordcnts)
        self.beta = beta_t
        if (self.type == 0 or self.type == 2):
            self.beta_drop = softmax_convert(self.rate * beta_t)
        if (self.type == 1):
            self.beta_drop = softmax_convert(beta_t)
