import numpy as np
from numpy.random import seed
from scipy.special import digamma
import time
from numpy import linalg as LA
import torch
import torch.nn as nn

seed(1)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def softmax_convert(matrix):
    """
    """
    matrix_max = torch.max(matrix, dim = 1)[0].view(-1,1)
    temp = torch.exp(matrix - matrix_max)
    matrix_softmax = temp / torch.sum(temp, dim=1).view(-1,1)
    return matrix_softmax

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return digamma(alpha) - digamma(sum(alpha))
    return (digamma(alpha) - digamma(np.sum(alpha, axis=1))[:, np.newaxis])


def softmax_convert_np(matrix):
    
    matrix_max = np.max(matrix, axis = 1)[:, np.newaxis]
    temp = np.exp(matrix - matrix_max)
    matrix_softmax = temp / np.sum(temp, axis=1)[:, np.newaxis]
    return matrix_softmax

def sigmoid_func_np(x):
    return 1/(1+np.exp(-x))

class model():
    def __init__(self, num_topic, n_term, batch_size, n_infer, learning_rate, 
                alpha, sigma, rate, type_model, iters):

        self.num_topic = num_topic
        self.n_term = n_term
        self.batch_size = batch_size
        self.n_infer = n_infer
        self.iters = iters  # the number of loop for training beta
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.sigma = sigma
        self.rate = rate
        self.type = type_model
        self.beta = softmax_convert_np(np.random.rand(self.num_topic, self.n_term))
        self.beta_drop = softmax_convert_np(self.beta)



    def train_model(self, wordinds, wordcnts, minibatch):
        
        start = time.time()
        gamma = np.random.gamma(100., 1./100., (self.batch_size, self.num_topic))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        beta_t = np.copy(self.beta)
        beta_p = self.beta

        if (minibatch <= 5):
            exp_pi = 1
        else:
            exp_pi = self.rate
        # exp_pi = self.rate  

        p = exp_pi
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
        pi_t = torch.tensor(pi_t).type(torch.DoubleTensor).to(device)

        for i in range(10):
          
            if (self.type == 0 or self.type == 2):
                beta_drop_t = softmax_convert_np(exp_pi * beta_t)
            if (self.type == 1):
                beta_drop_t = softmax_convert_np(beta_t)

            # print ('Updating gamma ...')      
            for d in range(self.batch_size):
                inds = wordinds[d]
                cnts = wordcnts[d]

                gamma_d = np.ones(self.num_topic) * self.alpha + float(np.sum(cnts)) / self.num_topic
                expElogtheta_d = np.exp(dirichlet_expectation(gamma_d))
                beta_d = beta_drop_t[:, inds]

                for i in range(self.n_infer):
                    phi_d = expElogtheta_d * beta_d.transpose() + 1e-10
                    phi_d /= np.sum(phi_d, axis=1)[:, np.newaxis]
                    gamma_d = self.alpha + np.dot(cnts, phi_d)
                    expElogtheta_d = np.exp(dirichlet_expectation(gamma_d))
                gamma[d] = gamma_d
                expElogtheta[d] = expElogtheta_d

            # Compute Sum_phi_i
            sum_phi_i = self.compute_sum_phi_i(wordinds, wordcnts, expElogtheta, beta_drop_t)
            sum_phi_i = torch.tensor(sum_phi_i).to(device)

            # Training Beta
            beta_t = self.train_beta(beta_t, beta_p, sum_phi_i, minibatch, pi_t)

        end = time.time()
        print ('Total time update: %f' % (end - start))
        

        #Update stream
        self.beta = beta_t


        if (self.type == 0 or self.type == 2):
            self.beta_drop = softmax_convert_np(exp_pi * self.beta)
        if (self.type == 1):
            self.beta_drop = softmax_convert_np(self.beta)

        

    def compute_sum_phi_i(self, wordinds, wordcnts, expElogtheta, beta_drop_t):
        sum_phi = np.zeros((self.num_topic, 1)) # k-element is sum(phi_dnk, d=1...D, n=1...Nd)
        sum_phi_i = np.zeros((self.num_topic, self.n_term)) # kj-element is sum(I[w_dn=j]*phi_dnk, d=1...D, n=1...Nd)

        for d in range(self.batch_size):
            inds = wordinds[d]
            cnts = wordcnts[d]
            expElogtheta_d = expElogtheta[d]
            beta_d = beta_drop_t[:, inds]

            phi_d = expElogtheta_d * beta_d.transpose() + 1e-10 # infer phi_d one more time
            phi_d /= np.sum(phi_d, axis=1)[:, np.newaxis] # shape len(ids)xK

            temp = cnts * phi_d.transpose()
            sum_phi += np.sum(temp, axis = 1)[:, np.newaxis]
            sum_phi_i[:, inds] += phi_d.transpose() * cnts

        return sum_phi_i


    def compute_loss(self, beta_t, beta_p, sum_phi_i, minibatch, pi_t):
        """
        Calculate loss function 

        Arguments:
            beta_t: beta in current task with shape (num_topic, num_terms)
            beta_p: beta in last previous task with shape (num_topic, num_terms)
            sum_phi_i: each kj-element in sum_phi_i is sum(I[w_dn=j]*phi_dnk, d=1...D, n=1...Nd)
        """

        beta_hat = beta_t*pi_t
        beta_hat = softmax_convert(beta_hat)
        log_beta_hat = torch.log(beta_hat)
        likelihood = (sum_phi_i*log_beta_hat).sum()
        reg = ((beta_t - beta_p)**2).sum()/(2*self.sigma)   # self.sigma is variance
        loss = -(likelihood - reg)
        return loss/self.batch_size




    def train_beta(self, beta_t, beta_p, sum_phi_i, minibatch, pi_t):
        """
        Training beta with fixed theta

        Arguments:
            beta_t: beta in current task with shape (num_topic, num_terms)
            beta_p: beta in last previous task with shape (num_topic, num_terms)
            theta_t: theta in current task with shape (num_topic, num_terms)
            theta_p: prior for pi in current task with shape (num_topic, num_terms)
            sum_phi_i: each kj-element in sum_phi_i is sum(I[w_dn=j]*phi_dnk, d=1...D, n=1...Nd)
        """
        
        beta_learnt = torch.tensor(beta_t).to(device).requires_grad_(True)  #make beta  becomes to a variable needed to grad
        beta_p = torch.tensor(beta_p).to(device)

        optimizer = torch.optim.Adagrad([beta_learnt], lr = self.learning_rate)

        for i in range(self.iters):
            optimizer.zero_grad()
            criterion = self.compute_loss(beta_learnt, beta_p, sum_phi_i, minibatch, pi_t)
            criterion.backward(retain_graph = True)
            if(i == self.iters-1):
                print("loss: ", criterion)
            optimizer.step()

            # Trick in concept drift model. But in default it is not used
            pi_t = np.random.binomial(1, self.rate, size=(self.num_topic, self.n_term))
            pi_t = torch.tensor(pi_t).type(torch.DoubleTensor).to(device)
            
        return beta_learnt.clone().detach().cpu().numpy()



        

        




    
        
        
