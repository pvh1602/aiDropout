import numpy as np
from numpy.random import seed
import sys, os, shutil
# import dropout_model
import utilities
import model_LDA
import time

all_dataset_dir = '../datasets'

all_model_folder = 'result_dropout'

def main():
	# check input
	if len(sys.argv) != 9:
		print ('Usage: python run_dropout.py [sigma] [rate] [times] [type_model] [num_sampling] [temperature] [epoches] [iters]')
		exit()

	sigma = float(sys.argv[1])
	rate = float(sys.argv[2])
	times = sys.argv[3]
	type_model = sys.argv[4]
	num_sampling = int(sys.argv[5])
	tau = float(sys.argv[6])
	epoches = int(sys.argv[7])
	iters = int(sys.argv[8])

	all_model_folder = 'result_dropout_epoch' + str(epoches) + '_iter' + str(iters)
	train_file = '/'.join([all_dataset_dir, 'all-train.txt'])
	setting_file = '/'.join([all_dataset_dir, 'new_setting.txt'])

	print ('Reading setting...')
	setting = utilities.read_setting(setting_file)
	num_topic = setting['num_topic']
	batch_size = setting['batch_size']
	
	n_term = setting['n_term']
	n_infer = setting['n_infer']
	learning_rate = setting['learning_rate']
	alpha = setting['alpha']
	mean = None
	
	if (type_model == 'B'):
		model_name = 'bernoulli'
	elif (type_model == 'S'):
		model_name = 'standard'
	elif (type_model == 'Z'):
		model_name = 'init_zero'
	else:
		print ('Unknown type model !!!')
		exit()

	sigma_folder = 'sigma' + str(sigma)
	model_folder = '/'.join([all_model_folder, sigma_folder, model_name])
	part_result = '/'.join([model_folder, 'result_' + times])
	GDS_result = '/'.join([part_result, 'GDS-' + str(batch_size) + '-' + str(rate)])	
	if os.path.exists(GDS_result):
		shutil.rmtree(GDS_result)
	os.makedirs(GDS_result)
	

	if (type_model == 'B'):
		MODEL = model_LDA.model(num_topic, n_term, batch_size, n_infer, learning_rate, alpha, sigma, mean, rate, tau, num_sampling, 0, epoches, iters)
	elif (type_model == 'S'):
		MODEL = model_LDA.model(num_topic, n_term, batch_size, n_infer, learning_rate, alpha, sigma, mean, rate, tau, num_sampling, 1, epoches, iters)
	elif (type_model == 'Z'):
		MODEL = model_LDA.model(num_topic, n_term, batch_size, n_infer, learning_rate, alpha, sigma, mean, rate, tau, num_sampling, 2, epoches, iters)
	else:
		print ('Unknown type model !!!')
		exit()

	f_train = open(train_file, 'r')
	perplexity_file = '/'.join([GDS_result, 'perplexities.csv'])
	(wordinds, wordcnts, stop) = utilities.read_minibatch_list_frequencies(f_train, batch_size)
	mini_batch = 0
	PPL = []

	while True:
		mini_batch += 1
		print ('[MINIBATCH %d]' % mini_batch)
		MODEL.train_model(wordinds, wordcnts, mini_batch)
		beta_minibatch = MODEL.beta_drop

		(wordinds, wordcnts, stop) = utilities.read_minibatch_list_frequencies(f_train, batch_size)
		if stop == 1:
			break
		wordinds1, wordcnts1, wordinds2, wordcnts2 = utilities.split_data_for_perplex(wordinds, wordcnts)

		print ('Compute perplexity...')
		s_time = time.time()
		(LD, ld2) = utilities.compute_perplexity(wordinds1, wordcnts1, wordinds2, wordcnts2, num_topic, n_term, n_infer, alpha, beta_minibatch)
		print ('Minibatch ' + str(mini_batch) + ' |=====| PERPLEXITY : ' +  ' sigma' + str(sigma) + ' rate' + str(rate) + ' : ' + str(LD) +  '\n')
		PPL.append(LD)
		print("Compute perplexity time: " + str(time.time()- s_time))

		# print perplexity
		utilities.write_perplexities(LD, perplexity_file)

		# print list_top each minibatch
		list_tops_minibatch = utilities.list_top(beta_minibatch, 20)
		list_tops_minibatch_file = '/'.join([GDS_result, 'list_tops_' + str(mini_batch) + '.dat'])
		utilities.write_topic_top(list_tops_minibatch, list_tops_minibatch_file)

	print(PPL)

	# print beta
	beta_final = MODEL.beta_drop
	beta_file = '/'.join([GDS_result, 'beta_final_' + times + '.txt'])		
	utilities.write_topics(beta_final, beta_file)

	# print list_top
	list_tops_final = utilities.list_top(beta_final, 20)
	list_tops_final_file = '/'.join([GDS_result, 'list_tops_final' + '.txt'])
	utilities.write_topic_top(list_tops_final, list_tops_final_file)

	# print top word
	top_word_file = '/'.join([GDS_result, 'top_word.txt'])
	utilities.write_top_word(list_tops_final, vocab_file, top_word_file)

	f_train.close()


if __name__ == '__main__':
	main()
