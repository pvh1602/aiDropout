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
                alpha, sigma, mean, prior_p, temperature, num_sampling, type_model, epoches, iters):
        self.num_topic = num_topic
        self.n_term = n_term
        self.batch_size = batch_size
        self.epoches = epoches     #if using coordinate GD
        self.n_infer = n_infer
        self.learning_rate = learning_rate
        self.iters = iters
        self.num_sampling = num_sampling
        self.alpha = alpha
        self.sigma = sigma
        self.mean = mean

        self.type = type_model
        self.temperature = temperature
        self.beta = softmax_convert_np(np.random.rand(self.num_topic, self.n_term))
        self.beta_drop = softmax_convert_np(self.beta)
        self.prior_p = np.ones((self.num_topic, self.n_term))*prior_p
        self.theta = np.random.rand(self.num_topic, self.n_term)   #theta is a matrix KxV



    def sample_pi(self, theta):

        """
        theta: is a probability which make variable get value 1, theta belongs to [0,1]
        
        return a sample which is relaxed by gumbel softmax trick
        """

        g1 = torch.Tensor(self.num_topic, self.n_term).uniform_(0,1).type(torch.DoubleTensor).to(device)
        g2 = torch.Tensor(self.num_topic, self.n_term).uniform_(0,1).type(torch.DoubleTensor).to(device)
        g1 = -torch.log(-torch.log(g1 + 1e-20)+1e-20)
        g2 = -torch.log(-torch.log(g2 + 1e-20)+1e-20)

        comp1 = (torch.log(theta) + g1)/self.temperature
        comp2 = (torch.log(1- theta) + g2)/self.temperature

        combined = torch.cat((comp1.unsqueeze(2), comp2.unsqueeze(2)), dim = 2) 
        max_comp = torch.max(combined, dim = 2)[0]
        pi_t = torch.exp(comp1 - max_comp)/ (torch.exp(comp2-max_comp) + torch.exp(comp1 - max_comp))

        return pi_t


    def sample_gumbel(self):
        g = torch.Tensor(self.num_topic, self.n_term).uniform_(0,1).type(torch.DoubleTensor).to(device)
        return -torch.log(-torch.log(g + 1e-20)+1e-20)
    

    def gumbel_softmax(self, theta, g1, g2):

        comp1 = (torch.log(theta) + g1)/self.temperature
        comp2 = (torch.log(1- theta) + g2)/self.temperature

        combined = torch.cat((comp1.unsqueeze(2), comp2.unsqueeze(2)), dim = 2) 
        max_comp = torch.max(combined, dim = 2)[0]
        pi_t = torch.exp(comp1 - max_comp)/ (torch.exp(comp2-max_comp) + torch.exp(comp1 - max_comp))

        return pi_t



    def compute_exp_pi(self, theta):
        """
        theta: is a probability which make variable get value 1, theta belongs to [0,1]
        
        return expectation of sample in gumbel softmax trick
        """
        num_exp_pi = np.exp(np.log(theta)/self.temperature)
        den_exp_pi = np.exp(np.log(1 - theta)/self.temperature) + np.exp(np.log(theta)/self.temperature)
        return num_exp_pi/den_exp_pi


    def train_model(self, wordinds, wordcnts, minibatch):

        gamma = np.random.gamma(100., 1./100., (self.batch_size, self.num_topic))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        beta_t = np.copy(self.beta)
        beta_p = self.beta
        theta_t = np.copy(self.theta) #theta_t = theta_t-1
        #dynamic theta
        # if minibatch == 1:
        #     theta_p = self.prior_p  
        # else:
        #     theta_p = np.copy(self.theta)

        #Learn prior_p
        theta_p = np.copy(self.prior_p)     #prior_t khởi tạo bằng prior = 0.5

        start = time.time()
        if (minibatch <= 5):
            exp_pi = 1
        else:
            # exp_pi = self.compute_exp_pi(sigmoid_func_np(theta_t))
            exp_pi = sigmoid_func_np(theta_t)
            # exp_pi = 1

        g1 = self.sample_gumbel()
        g2 = self.sample_gumbel()

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


            sum_phi_i = self.compute_sum_phi_i(wordinds, wordcnts, expElogtheta, beta_drop_t)
            sum_phi_i = torch.tensor(sum_phi_i).to(device)
            
            # beta_t = self.train_beta1(beta_t, beta_p, sum_phi_i)
            # beta_t, theta_t = self.train_both(beta_t, beta_p, theta_t, theta_p, sum_phi_i)

            #Train model coordinate gradient descent
            for epoch in range(self.epoches):
                
                beta_t = self.train_beta(beta_t, beta_p, theta_t, theta_p, sum_phi_i, epoch, minibatch, g1, g2)
                if minibatch > 5:
                    theta_t, theta_p = self.train_theta(beta_t, beta_p, theta_t, theta_p, sum_phi_i, epoch, minibatch,g1,g2)

        end = time.time()
        print ('Total time update: %f' % (end - start))
        

        #Update stream
        self.beta = beta_t
        self.theta = theta_t
        # self.prior_p = theta_p
        # print("beta: ", beta_t)
        # print("theta: ", theta_t)

        if (minibatch <= 5):
            exp_pi = 1
        else:
            # exp_pi = self.compute_exp_pi(sigmoid_func_np(self.theta))
            exp_pi = sigmoid_func_np(theta_t)
            # exp_pi = 1
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


    def compute_loss(self, beta_t, beta_p, theta_t, theta_p, sum_phi_i, minibatch, g1, g2):
        """
        Calculate loss function 

        Arguments:
            beta_t: beta in current task with shape (num_topic, num_terms)
            beta_p: beta in last previous task with shape (num_topic, num_terms)
            theta_t: theta in current task with shape (num_topic, num_terms)
            theta_p: theta in previous task with shape (num_topic, num_terms)
            sum_phi_i: each kj-element in sum_phi_i is sum(I[w_dn=j]*phi_dnk, d=1...D, n=1...Nd)
        """

        #the first five minibatch data is not dropped
        if minibatch <= 5:
            pi_t = 1
            beta_hat = beta_t*pi_t
            beta_hat = softmax_convert(beta_hat)
            log_beta_hat = torch.log(beta_hat)
            likelihood = (sum_phi_i*log_beta_hat).sum()
            reg = ((beta_t - beta_p)**2).sum()/(2*self.sigma)
            loss = -(likelihood - reg)
            return loss/self.batch_size

        else:
            likelihood = 0
            for i in range(self.num_sampling):
                pi_t = self.gumbel_softmax(theta_t, g1, g2)
                # print("beta_t: ", beta_t)
                # print("pi_t_t: ", pi_t)
                beta_hat = beta_t*pi_t
                beta_hat = softmax_convert(beta_hat)
                log_beta_hat = torch.log(beta_hat)

                likelihood += (sum_phi_i*log_beta_hat).sum() / self.num_sampling
            kl_bern = self.compute_KL_bernoulli(theta_t, theta_p).sum()
            reg = ((beta_t - beta_p)**2).sum()/(2*self.sigma)
            loss = -(likelihood - kl_bern - reg)


            return loss / self.batch_size



    def compute_KL_bernoulli(self, theta1, theta2):
        """
        theta: is a probability which make variable get value 1 in Bernoulli distribution, theta belongs to [0,1]
        
        Return KL divergence between two Bernoulli distribution
        """
        return theta1*torch.log(theta1/theta2) + (1-theta1)*torch.log((1-theta1)/(1-theta2))




    def train_beta(self, beta_t, beta_p, theta_t, theta_p, sum_phi_i, epoch, minibatch, g1, g2):
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
        theta_t = torch.tensor(theta_t).to(device)
        theta_t = 1/(1+torch.exp(-theta_t))
        theta_p = torch.tensor(theta_p).to(device)     #if theta_p = prior_P
        theta_p = 1/(1+torch.exp(-theta_p))            #sigmoid(theta)

        optimizer = torch.optim.Adagrad([beta_learnt], lr = self.learning_rate)

        for i in range(self.iters):
            optimizer.zero_grad()
            criterion = self.compute_loss(beta_learnt, beta_p, theta_t, theta_p, sum_phi_i, minibatch, g1, g2)
            criterion.backward(retain_graph = True)
            if(epoch == self.epoches-1 and i == self.iters-1):
                print("loss_beta: ", criterion)
            optimizer.step()
            
        
        return beta_learnt.clone().detach().cpu().numpy()





    def train_theta(self, beta_t, beta_p, theta_t, theta_p, sum_phi_i, epoch, minibatch, g1, g2):
        """
        Training theta with fixed beta

        Arguments:
            beta_t: beta in current task with shape (num_topic, num_terms)
            beta_p: beta in last previous task with shape (num_topic, num_terms)
            theta_t: theta in current task with shape (num_topic, num_terms)
            theta_p: theta in previous task with shape (num_topic, num_terms)
            sum_phi_i: each kj-element in sum_phi_i is sum(I[w_dn=j]*phi_dnk, d=1...D, n=1...Nd)
        """
        beta_t = torch.tensor(beta_t).to(device)
        beta_p = torch.tensor(beta_p).to(device)
        theta_learnt = torch.tensor(theta_t).to(device).requires_grad_(True) 
        theta_p_learnt = torch.tensor(theta_p).to(device).requires_grad_(True)
         #if theta_p = theta in task t-1

        optimizer = torch.optim.Adagrad([theta_learnt, theta_p_learnt], lr = self.learning_rate)

        for i in range(self.iters):
            optimizer.zero_grad()

            theta_learnt_sig = 1/(1+torch.exp(-theta_learnt)) #cast theta in range [0,1] by sigmoid function
            theta_p_learnt_sig = 1/(1+torch.exp(-theta_p_learnt))  #cast theta_p in range [0,1] by sigmoid function

            criterion = self.compute_loss(beta_t, beta_p, theta_learnt_sig, theta_p_learnt_sig, sum_phi_i, minibatch, g1, g2)
            criterion.backward(retain_graph = True)
            if(epoch == self.epoches-1 and i == self.iters-1):
                print("loss_theta: ", criterion)
            optimizer.step()
            
        
        return theta_learnt.clone().detach().cpu().numpy(), theta_p_learnt.clone().detach().cpu().numpy()




    def train_both(self, beta_t, beta_p, theta_t, theta_p, sum_phi_i):
        """
        Training with both beta and theta are variables

        Arguments:
            beta_t: beta in current task with shape (num_topic, num_terms)
            beta_p: beta in last previous task with shape (num_topic, num_terms)
            theta_t: theta in current task with shape (num_topic, num_terms)
            theta_p: theta in previous task with shape (num_topic, num_terms)
            sum_phi_i: each kj-element in sum_phi_i is sum(I[w_dn=j]*phi_dnk, d=1...D, n=1...Nd)
        """


        #Convert theta, beta into cuda and make beta_t, theta_t become dynamic nodes
        beta_learnt = torch.tensor(beta_t).to(device).requires_grad_(True)
        theta_learnt = torch.tensor(theta_t).to(device).requires_grad_(True)
        beta_p = torch.tensor(beta_p).to(device)
        theta_p = torch.tensor(theta_p).to(device)    #if theta_p = prior_P
        # theta_p = 1/(1+torch.exp(-theta_p)) #if theta_p = theta in task t-1

        #define optimizer 
        optimizer = torch.optim.Adagrad([beta_learnt, theta_learnt], lr = self.learning_rate)
        for i in range(self.iters):
            optimizer.zero_grad()
            theta_learnt_sig = 1/(1+torch.exp(-theta_learnt)) #cast theta in range [0,1] by sigmoid function
            criterion = self.compute_loss(beta_learnt, beta_p, theta_learnt_sig, theta_p, sum_phi_i)
            criterion.backward(retain_graph = True)
            optimizer.step()
        beta_t = beta_learnt.clone().detach().cpu().numpy()
        theta_t = theta_learnt.clone().detach().cpu().numpy()

        return beta_t, theta_t




   

        

        




    
        
        
