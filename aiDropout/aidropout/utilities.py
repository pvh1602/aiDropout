import numpy as np
import string
import sys
import perplexity
# from scipy.special import digama

"""
	Read all documents in the file and stores terms and counts in lists.
"""
def read_data(filename):
	wordinds = list()
	wordcnts = list()
	fp = open(filename, 'r')
	while True:
		line = fp.readline() # string '6 2:1 571:1 9:1 569:1 123:2 572:1'
		# check end of file
		if len(line) < 1:
			break
		terms = str.split(line) # ['6', '2:1', '571:1', '9:1', '569:1', '123:2', '572:1']
		doc_length = int(terms[0])
		inds = np.zeros(doc_length, dtype = np.int32)
		cnts = np.zeros(doc_length, dtype = np.int32)
		for j in range(1, doc_length + 1):
			term_count = terms[j].split(':') # such as ['2', '1']
			inds[j - 1] = int(term_count[0])
			cnts[j - 1] = int(term_count[1])
		wordinds.append(inds) # A list D elements, each element is an array such as array([  2, 571,   9, 569, 123, 572])
		wordcnts.append(cnts)
	fp.close()
	return (wordinds, wordcnts)

"""
	Read data for computing perplexities.
"""
def read_data_for_perplex(dataset_dir, num_test):
	data_test = list()
	wordinds1 = list()
	wordcnts1 = list()
	wordinds2 = list()
	wordcnts2 = list()
	for i in range(num_test):
		filename_part1 = '%s/data_test_%d_part_1.txt'%(dataset_dir, i+1)
		filename_part2 = '%s/data_test_%d_part_2.txt'%(dataset_dir, i+1)

		(wordinds_1, wordcnts_1) = read_data(filename_part1)
		(wordinds_2, wordcnts_2) = read_data(filename_part2)

		wordinds1.append(wordinds_1)
		wordcnts1.append(wordcnts_1)
		wordinds2.append(wordinds_2)
		wordcnts2.append(wordcnts_2)

	return wordinds1, wordcnts1, wordinds2, wordcnts2

def split_data_for_perplex(wordinds, wordcnts):
    wordinds1 = list()
    wordcnts1 = list()
    wordinds2 = list()
    wordcnts2 = list()
    for d in range(len(wordinds)):
        l = len(wordinds[d])
        if (l < 5):
            wordinds1.append(np.split(wordinds[d], [-1])[0])
            wordcnts1.append(np.split(wordcnts[d], [-1])[0])
            wordinds2.append(np.split(wordinds[d], [-1])[1])
            wordcnts2.append(np.split(wordcnts[d], [-1])[1])

        else:
            pivot = int(np.floor(l * 4.0 /5))
            wordinds1.append(np.split(wordinds[d], [pivot])[0])
            wordcnts1.append(np.split(wordcnts[d], [pivot])[0])
            wordinds2.append(np.split(wordinds[d], [pivot])[1])
            wordcnts2.append(np.split(wordcnts[d], [pivot])[1])

    return wordinds1, wordcnts1, wordinds2, wordcnts2

def compute_perplexity(wordinds1, wordcnts1, wordinds2, wordcnts2, num_topic, n_term, n_infer, alpha, beta):
	perplex = perplexity.Stream(alpha, beta, num_topic, n_term, n_infer)
	LD = 0
	ld2 = list()
	ld = perplex.compute_perplexity(wordinds1, wordcnts1, wordinds2, wordcnts2)
	LD += ld
	ld2.append(ld)
	return (LD, ld2)

"""
	Read mini-batch and stores terms and counts in lists
"""
def read_minibatch_list_frequencies(fp, batch_size):
    wordinds = list()
    wordcnts = list()
    stop = 0
    for i in range(batch_size):
        line = fp.readline()
        # check end of file
        if len(line) < 1:
            stop = 1
            break

        terms = str.split(line)
        # print (terms)
        doc_length = len(terms)
        inds = np.zeros(doc_length, dtype = np.int32)
        cnts = np.zeros(doc_length, dtype = np.int32)
        for j in range(0, doc_length):
            term_count = terms[j].split(':')
            inds[j] = int(term_count[0])
            cnts[j] = int(term_count[1])
        wordinds.append(inds)
        wordcnts.append(cnts)
    return (wordinds, wordcnts, stop)

"""
	Read mini-batch and stores each document as a sequence of tokens (wordtks: token1 token2 ...).
"""
def read_minibatch_list_sequences(fp, batch_size):
	wordtks = list()
	lengths = list()

	for i in range(batch_size):
		line = fp.readline()
		# check end of file
		if len(line) < 1:
			break
		tks = list()
		tokens = str.split(line)
		counts = int(tokens[0])
		for j in range(1, counts + 1):
			token_count = tokens[j].split(':')
			token_count = list(map(int, token_count)) # such as [2, 1]
			for k in range(token_count[1]): # replace = TRUE
				tks.append(token_count[0]) 
		wordtks.append(tks)  # a list such as [2, 571, 9, 569, 123, 123, 572], including repeating elements
		lengths.append(len(tks)) # number element, including repeating
	return (wordtks, lengths)



"""
	Read mini-batch and stores in dictionary (train_cts: (term:frequency)).
"""
def read_minibatch_dict(fp, batch_size):
	train_cts = list()
	stop = 0
	for i in range(batch_size):
		line = fp.readline()
		# check end of file
		if len(line) < 5:
			stop = 1
			break
		ids = list()
		cts = list()
		terms = str.split(line)
		doc_length = int(terms[0])
		for j in range(1, doc_length + 1):
			term_count = terms[j].split(':')
			ids.append(int(term_count[0]))
			cts.append(int(term_count[1]))
		ddict = dict(zip(ids, cts)) # such as {2: 1, 9: 1, 123: 2, 569: 1, 571: 1, 572: 1}
		train_cts.append(ddict) # a list of dictionaries
	return(train_cts, stop)

"""
    Read setting file.
"""    
def read_setting(file_name):
    f = open(file_name, 'r')
    settings = f.readlines() # a list of lines of setting file, each element is a string - line
    f.close()
    sets = list()
    vals = list()
    for i in range(len(settings) - 1):
        if len(settings[i]) < 2:
            break
        set_val = settings[i].split(':')
        # print (set_val)
        sets.append(set_val[0])
        vals.append(float(set_val[1]))
    ddict = dict(zip(sets, vals))
    ddict['n_term'] = int(ddict['n_term'])
    ddict['num_topic'] = int(ddict['num_topic'])
    ddict['batch_size'] = int(ddict['batch_size'])
    ddict['n_infer'] = int(ddict['n_infer'])
    ddict['num_test'] = int(ddict['num_test'])
    return (ddict)
 
"""
    Create list of top words of topics.
"""
def list_top(beta, ntops):
    # list_tops = list()
    # for i in range(len(beta)):
    #     tmp = beta[i].argsort()[-10:][::-1]
    #     list_tops.append(list(tmp))

    min_float = -sys.float_info.max
    num_topic = beta.shape[0]
    list_tops = list()
    for k in range(num_topic):
        top = list() 
        arr = np.array(beta[k,:], copy = True)
        for t in range(ntops):
            index = arr.argmax()
            top.append(index)
            arr[index] = min_float
        list_tops.append(top)
    return (list_tops) 

def write_top_word(list_tops, vocab_file, top_word_file):
    # get the vocabulary
    vocab = open(vocab_file, 'r').readlines()
    vocab = list(map(lambda x: x.strip(), vocab))
    # print ('Vocabulary example:', vocab[:10])
    topic_no = 1
    # write to file
    f = open(top_word_file, 'w+')

    for list_top in list_tops:
        f.write('TOPIC '+ str(topic_no) + ': ')
        for i in range(len(list_top)):
            # f.write(vocab[list_top[i]] + ' : ' + str(list_top[i]) + '\n')
            f.write(vocab[list_top[i]] + ' ')
        f.write('\n')
        topic_no += 1
    f.close()

"""
-------------------------------------------------------------------------------
"""   
                
def write_topics(beta, file_name):
    num_terms = beta.shape[1]
    num_topic = beta.shape[0]
    f = open(file_name, 'w')
    for k in range(num_topic):
        for i in range(num_terms - 1):
            f.write('%.10f '%(beta[k][i]))
        f.write('%.10f\n'%(beta[k][num_terms - 1]))
    f.close()

def write_topic_mixtures(theta, file_name):
    batch_size = theta.shape[0]
    num_topic = theta.shape[1]
    f = open(file_name, 'a')
    for d in range(batch_size):
        for k in range(num_topic - 1):
            f.write('%.5f '%(theta[d][k]))
        f.write('%.5f\n'%(theta[d][num_topic - 1]))
    f.close()

def write_perplexities(LD, file_name):
    f = open(file_name, 'a')
    f.write('%f,'%(LD))
    # for i in range(len(LD)):
    #     f.write('%f,'%(LD[i]))
    f.close()

def write_topic_top(list_tops, file_name):
    num_topic = len(list_tops)
    tops = len(list_tops[0])
    f = open(file_name, 'w')
    for k in range(num_topic):
        for j in range(tops - 1):
            f.write('%d '%(list_tops[k][j]))
        f.write('%d\n'%(list_tops[k][tops - 1]))
    f.close()

def write_sparsity(sparsity, file_name):
    f = open(file_name, 'a')
    f.write('%.10f,' % (sparsity))
    f.close()
    
def write_time(i, j, time_e, time_m, file_name):
    f = open(file_name, 'a')
    f.write('tloop_%d_iloop_%d, %f, %f, %f,\n'%(i, j, time_e, time_m, time_e + time_m))
    f.close()
    
def write_loop(i, j, file_name):
    f = open(file_name, 'w')
    f.write('%d, %d'%(i,j))
    f.close()
    
def write_setting(ddict, file_name):
    keys = ddict.keys()
    vals = ddict.values()
    f = open(file_name, 'w')
    for i in range(len(keys)):
        f.write('%s: %f\n'%(keys[i], vals[i]))
    f.close()

def write_file(i, j, beta, time_e, time_m, theta, sparsity, LD2, list_tops, tops, model_folder):
    beta_file_name = '%s/beta_%d_%d.dat'%(model_folder, i, j)
    theta_file_name = '%s/theta_%d.dat'%(model_folder, i)
    per_file_name = '%s/perplexities_%d.csv'%(model_folder, i)
    top_file_name = '%s/top%d_%d_%d.dat'%(model_folder, tops, i, j)
    spar_file_name = '%s/sparsity_%d.csv'%(model_folder, i)
    time_file_name = '%s/time_%d.csv'%(model_folder, i)
    loop_file_name = '%s/loops.csv'%(model_folder)
    """    
    # write beta    
    if j % 10 == 1:
        write_topics(beta, beta_file_name)
    # write theta
    write_topic_mixtures(theta, theta_file_name)
    """
    # write perplexities
    write_perplexities(LD2, per_file_name)
    # write list top
    ##write_topic_top(list_tops, top_file_name)
    # write sparsity
    ##write_sparsity(sparsity, spar_file_name)
    # write time
    write_time(i, j, time_e, time_m, time_file_name)
    # write loop
    write_loop(i, j, loop_file_name)