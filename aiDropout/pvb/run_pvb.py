import numpy as np
import sys, os, shutil
import model_pvb
import utilities
from shutil import copyfile


all_dataset_dir = '../datasets'
model_folder = 'result_pvb'

def main():
	# check input
	if len(sys.argv) != 5:
		print ('Usage: python run.py [tau0] [kappa] [M] [times]')
		exit()

	tau0 = float(sys.argv[1])
	kappa = float(sys.argv[2])
	M = int(sys.argv[3])
	times = sys.argv[4]

	train_file = '/'.join([all_dataset_dir, 'all-train.txt'])
	setting_file = '/'.join([all_dataset_dir, 'new_setting.txt'])
	vocab_file = '/'.join([all_dataset_dir, 'vocab.txt'])

	print ('Reading setting...')
	setting = utilities.read_setting(setting_file)
	num_topic = setting['num_topic']
	batch_size = setting['batch_size']
	
	n_term = setting['n_term']
	n_infer = setting['n_infer']
	learning_rate = setting['learning_rate']
	alpha = setting['alpha']
	eta = 0.01

	data_result = '/'.join([model_folder, 'tau_' + str(tau0), 'kappa_' + str(kappa), 'M_' + str(M)])
	if not os.path.exists(data_result):
		os.makedirs(data_result)

	MODEL = model_pvb.Model(num_topic, n_term, batch_size, n_infer, alpha, eta, tau0, kappa, M)

	f_train = open(train_file, 'r')
	perplexity_file = '/'.join([data_result, 'perplexities.csv'])
	(wordinds, wordcnts, stop) = utilities.read_minibatch_list_frequencies(f_train, batch_size)
	mini_batch = 0
	PPL = []

	while True:
		mini_batch += 1
		print ('[MINIBATCH %d]' % mini_batch)
		MODEL.update_stream(wordinds, wordcnts, mini_batch)
		lamda_minibatch = MODEL.lamda

		(wordinds, wordcnts, stop) = utilities.read_minibatch_list_frequencies(f_train, batch_size)
		if stop == 1:
			break
		wordinds1, wordcnts1, wordinds2, wordcnts2 = utilities.split_data_for_perplex(wordinds, wordcnts)

		print ('Compute perplexity...')
		(LD, ld2) = utilities.compute_perplexity(wordinds1, wordcnts1, wordinds2, wordcnts2, num_topic, n_term, n_infer, alpha, lamda_minibatch)
		print ('Minibatch ' + str(mini_batch) + ' |=====| PERPLEXITY : ' + str(LD) +  '\n')
		PPL.append(LD)

		# print perplexity
		utilities.write_perplexities(LD, perplexity_file)

		# print list_top each mini_batch
		beta_minibatch = lamda_minibatch / np.sum(lamda_minibatch, axis=1)[:, np.newaxis]
		list_tops_minibatch = utilities.list_top(beta_minibatch, 20)
		list_tops_minibatch_file = '/'.join([data_result, 'list_tops_' + str(mini_batch) + '.dat'])
		utilities.write_topic_top(list_tops_minibatch, list_tops_minibatch_file)

	print(PPL)

	# print beta
	lamda_final = MODEL.lamda
	beta_final = lamda_final / np.sum(lamda_final, axis = 1)[:, np.newaxis]
	beta_file = '/'.join([data_result, 'beta_final_' + times + '.txt'])		
	utilities.write_topics(beta_final, beta_file)

	# print list_top
	list_tops_final = utilities.list_top(beta_final, 20)
	list_tops_final_file = '/'.join([data_result, 'list_tops_final' + '.txt'])
	utilities.write_topic_top(list_tops_final, list_tops_final_file)

	# print top word
	top_word_file = '/'.join([data_result, 'top_word.txt'])
	utilities.write_top_word(list_tops_final, vocab_file, top_word_file)

	f_train.close()

if __name__ == '__main__':
	main()

