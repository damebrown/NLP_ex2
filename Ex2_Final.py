import copy
import re
import random
from nltk.corpus import brown

global tagged_train_set, tagged_test_set
word_tag_count, word_count, tag_count = {}, {}, {}
unknown_words = {}
pseudo_word_count = {}
UNKNOWN_TAG = 'NN'
REGEX_PATTERN = '^[^+-]*'
LOW_FREQ_CONSTANT = 4
WORD_INDEX = 0
TAG_INDEX = 1


def extract_unknown():
	"""
	this function puts all unknown words in the test set in the global dictionary "unknown_words"
	"""
	for sentence in tagged_test_set:
		for word, _ in sentence:
			if word not in word_count:
				if word in unknown_words:
					unknown_words[word] += 1
				else:
					unknown_words[word] = 1


#          4.a- load the data          #
def get_data():
	"""
	this function downloads the brown data set and sets the global variables tagged_train_set, tagged_test_set
	"""
	tagged_sen = brown.tagged_sents(categories = ['news'])
	split_var = int(0.9 * len(tagged_sen))
	global tagged_train_set, tagged_test_set
	tagged_train_set = tagged_sen[:split_var]
	tagged_test_set = tagged_sen[split_var:]


#          4.b- implement most likely tag          #
def count_appearances():
	"""
	This function counts the number of words appearance, number of tag appearances and number of word-tag appearances
	in the train set, and returns all the counts in a dictionary
	:return: word-tag count dictionary, word count dictionary, tag count dictionary
	"""
	for sent in tagged_train_set:
		for word, tag in sent:
			tag = simplify_tag(tag)
			if word not in word_count:
				word_count[word] = 1
			else:
				word_count[word] += 1
			if tag not in tag_count:
				tag_count[tag] = 1
			else:
				tag_count[tag] += 1
			if (word, tag) not in word_tag_count:
				word_tag_count[(word, tag)] = 1
			else:
				word_tag_count[(word, tag)] += 1


def ml_tag():
	"""
	This function implements the most likely tag baseline, as requested in 4.b.
	:return: total error (real number e s.t. 0<=e<=1), error for known words and error for unknown words
	"""
	word_tag_probabilities = compute_probabilities()
	# word_tag_probabilities is a dict of shape {(word,tag): probability of tag, given word}
	# compute_mlt computes the answer for 4.b.i, by computing the tag that maximizes p(tag|word)
	words_mlt = compute_mlt(word_tag_probabilities)
	# words_mlt is a dict of shape {word: (tag, probability)}
	known_test_results, unknown_test_results = compute_test_results(words_mlt)
	# test_results is a dict of shape {word: (# of appearances, # of correct predictions)}
	return compute_accuracy(known_test_results, unknown_test_results)


def simplify_tag(tag):
	"""
	this function returns the prefix of a tag if there is a '+' or '-' in it
	:param tag: the tag to simplify
	:return: the prefix of the tag if there is a '+' or '-' in it
	"""
	return re.findall("^[^+-]*", tag)[0]


def compute_test_results(words_mlt):
	"""
	This function computes the result of the MLT on the test set.
	:param words_mlt: a dict of shape {word: (tag, probability)}
	:return: known_test_results is a dict of shape {known_word: (# of appearances, # of correct predictions)}
			unknown_test_results is a dict of shape {unknown_word: (# of appearances, # of correct predictions)}
	"""
	known_test_results = {}
	# {word: (# of appearances, # of correct predictions)}
	unknown_test_results = {}
	# {word: (# of appearances, # of correct predictions)}
	for sent in tagged_test_set:
		for word, tag in sent:
			# if word is known:
			if word in words_mlt:
				if word not in known_test_results:
					known_test_results[word] = [1, 0]
				else:
					known_test_results[word][0] += 1
				if words_mlt[word][0] == tag:
					known_test_results[word][1] += 1
			# if word is unknown:
			else:
				if word not in unknown_test_results:
					unknown_test_results[word] = [1, 0]
				else:
					unknown_test_results[word][0] += 1
				if tag == UNKNOWN_TAG:
					unknown_test_results[word][1] += 1
	return known_test_results, unknown_test_results


def compute_mlt(word_tag_probabilities):
	words_mlt = {}  # {word: (tag, probability)}
	# find the tag that maximizes p(tag|word)
	for word, tag in word_tag_probabilities.keys():
		if word not in words_mlt:
			words_mlt[word] = (tag, word_tag_probabilities[(word, tag)])
		elif word_tag_probabilities[(word, tag)] > words_mlt[word][1]:
			words_mlt[word] = (tag, word_tag_probabilities[(word, tag)])
	return words_mlt


def compute_probabilities():
	"""
	computes the probabilities p(tag|word) for each word-tag tuple
	:return: the dict word_tag_probabilities of the shape {(word,tag): p(tag|word)}
	"""
	word_tag_probabilities = {}  # {(word,tag): p(tag|word)}
	# calculate the probability for each word-tag pair
	for word_tag in word_tag_count:
		word_tag_probabilities[word_tag] = word_tag_count[word_tag] / word_count[word_tag[WORD_INDEX]]
	return word_tag_probabilities


def compute_accuracy(known_test_results, unknown_test_results):
	"""
	receives the test results and computes the accuracy and error, for each word and for the total MLE
	:param known_test_results: a dict of shape {known_word: (# of appearances, # of correct predictions)}
	:param unknown_test_results: a dict of shape {unknown_word: (# of appearances, # of correct predictions)}
	:return: total error (real number e s.t. 0<=e<=1), error for known words and error for unknown words
	"""
	# compute error of known words
	known_corrects_counter = 0
	known_words_counter = 0
	for word in known_test_results:
		known_words_counter += known_test_results[word][0]
		known_corrects_counter += known_test_results[word][1]
	known_error = 1 - (known_corrects_counter / known_words_counter)

	# compute error of unknown words
	unknown_corrects_counter = 0
	unknown_words_counter = 0
	for word in unknown_test_results:
		unknown_words_counter += unknown_test_results[word][0]
		unknown_corrects_counter += unknown_test_results[word][1]
	unknown_error = 1 - (unknown_corrects_counter / unknown_words_counter)

	# compute total error
	total_corrects = known_corrects_counter + unknown_corrects_counter
	total_words = known_words_counter + unknown_words_counter
	final_error = 1 - (total_corrects / total_words)

	return known_error, unknown_error, final_error


#          4.c- implement a bi-gram HMM tagger         #
def e():
	"""
	this function computes the emission probability function
	:return: a dictionary of the shape {(word, tag): p(y_i|x_i)}
	"""
	_emission = copy.deepcopy(word_tag_count)
	for word_tag in word_tag_count:
		_emission[word_tag] /= tag_count[word_tag[TAG_INDEX]]
	return _emission


def q():
	transmission = {}  # a dict of shape {(tag_i, tag_j): probability}
	tag_count_ = copy.deepcopy(tag_count)

	tag_count_['START'], tag_count_['STOP'] = 0, len(tagged_train_set)

	# initialize all possible couples of tags
	for tag_i in tag_count_:
		if tag_i != 'START':
			for tag_j in tag_count_:
				if tag_j != 'STOP':
					transmission[(tag_i, tag_j)] = 0
	for sent in tagged_train_set:
		for i, word_n_tag in enumerate(sent):
			tag = simplify_tag(word_n_tag[TAG_INDEX])
			if i == 0:
				transmission[(tag, 'START')] += 1
			else:
				transmission[(tag, simplify_tag(sent[i - 1][TAG_INDEX]))] += 1
		transmission[('STOP', simplify_tag(sent[-1][TAG_INDEX]))] += 1

	for tagi_tagj in transmission:
		if transmission[tagi_tagj] == 0:
			transmission[tagi_tagj] = 10 ** (-24)
		transmission[tagi_tagj] /= tag_count_[tagi_tagj[0]]

	return transmission


def viterbi(sentence, q, e):
	# Todo consider unify with tags dictionary
	S = []
	for key in tag_count:
		S.append(key)

	n = len(sentence)
	s = len(S)
	pi = [[0 for i in range(s)] for j in range(n)]
	bp = [[0 for i in range(s)] for j in range(n)]

	# Initialize table first row
	for v in range(s):
		pi[0][v] = q[(S[v], 'START')]

	for k in range(1, n):
		for v in range(s):
			max_val = 0
			prev_tag_index = random.randint(0, s - 1)
			emission = 10 ** (-24)
			if (sentence[k][0], S[v]) in e:
				emission = e[(sentence[k][0], S[v])]
			for w in range(s):
				prev = pi[k - 1][w]
				transition = q[(S[v], S[w])]  # probability for v given w (tags indexes)
				if prev * transition > max_val:
					max_val = prev * transition
					prev_tag_index = w
			pi[k][v] = max_val * emission
			bp[k][v] = prev_tag_index

	# Final probability fot sentence
	final_max = 0
	for i in range(s):
		temp = pi[n - 1][i] * q[('STOP', S[i])]
		if temp > final_max:
			final_max = temp

	# Get tags chain from bp
	tags = [0 for i in range(n)]

	# Get last tag
	max_val = 0
	last_tag_index = 0
	for v in range(s):
		prev = pi[n - 1][v]
		transition = q[('STOP', S[v])]
		if prev * transition > max_val:
			max_val = prev * transition
			last_tag_index = v

	# Get all other tags
	tags[n - 1] = S[last_tag_index]
	cur = last_tag_index

	for i in range(n - 2, -1, -1):
		tags[i] = S[bp[i + 1][cur]]
		cur = bp[i + 1][cur]

	return tags


def viterbi_error_rate(emission_dict, confusion_flag = False):
	viterbi_res = []
	q_dict = q()
	for i in range(len(tagged_test_set)):
		viterbi_res.append(viterbi(tagged_test_set[i], q_dict, emission_dict))

	# count errors by known/unknown/general
	known_words_num_of_error_tags = 0
	unknown_words_num_of_error_tags = 0
	general_num_of_error_tags = 0

	unknown_words_counter = 0
	known_words_counter = 0

	confusion_dict = {}  # a dict of shape {(true_tag, falsely_predicted_tag): # of appearances}
	for i in range(len(viterbi_res)):
		for j in range(len(viterbi_res[i])):
			word = tagged_test_set[i][j][WORD_INDEX]
			true_tag = tagged_test_set[i][j][TAG_INDEX]
			if word in unknown_words.keys() or is_low_freq(word):
				unknown_words_counter += 1
			else:
				known_words_counter += 1
			if viterbi_res[i][j] != true_tag:
				general_num_of_error_tags += 1
				if word in train_words_after_pseudo_count:
					known_words_num_of_error_tags += 1
				else:
					unknown_words_num_of_error_tags += 1
					confusion_dict = update_confusion_dict(viterbi_res[i][j], true_tag, confusion_dict)
	if confusion_flag:
		return (known_words_num_of_error_tags / known_words_counter),\
				(unknown_words_num_of_error_tags / unknown_words_counter),\
				(general_num_of_error_tags / sum([len(sent) for sent in tagged_test_set])),\
				confusion_dict
	else:
		return (known_words_num_of_error_tags / known_words_counter),\
				(unknown_words_num_of_error_tags / unknown_words_counter),\
				(general_num_of_error_tags / sum([len(sent) for sent in tagged_test_set]))


def print_confusion(confusion_dict):
	confusion_dict_sorted = sorted(confusion_dict.items(), key=lambda item: item[1], reverse=True)
	print(confusion_dict_sorted)
	return


def update_confusion_dict(falsely_predicted_tag, true_tag, confusion_dict):
	if (true_tag, falsely_predicted_tag) in confusion_dict:
		confusion_dict[(true_tag, falsely_predicted_tag)] += 1
	else:
		confusion_dict[(true_tag, falsely_predicted_tag)] = 1
	return confusion_dict


#          4.d- Laplace add-one smoothing         #
def add1_e():
	add1_emission = {}
	for word in word_count.keys():
		for tag in tag_count:
			add1_denominator = (len(word_count) + tag_count[tag])
			if (word, tag) not in word_tag_count:
				add1_emission[(word, tag)] = 1 / add1_denominator
			else:
				# if (word, tag) exists, add 1:
				add1_emission[(word, tag)] = (word_tag_count[(word, tag)] + 1) / add1_denominator
	return add1_emission


#          4.e- Pseudo-words smoothing         #
def which_pseudo(word):
	"""
	this function receives as input a low frequency word and returns the corresponding pseudo word,
	set by the prior chosen design
	:param word: the low frequency word
	:return: the representing pseudo word
	"""
	if word.endswith("ING") or word.endswith("ing"):
		return "PRESENT_PERFECT"
	if word.endswith("ED") or word.endswith("ed"):
		return "PAST"
	elif '$' in word:
		return "DOLLAR"
	elif word.isdigit():
		return "DIGITS"
	elif word.isupper():
		return "CAPS"
	elif word[0].isupper():
		return "CAPS_PREFIX"
	elif '.' in word:
		return "CONTAINS_DOT"
	elif '/' in word or '\\' in word:
		return "CONTAINS_SLASH"
	elif '\'' in word:
		return "CONTAINS_APOSTROPHE"
	else:
		return "GENERIC_LOW_FREQUENCY"


def is_low_freq(word):
	return word_count[word] < LOW_FREQ_CONSTANT


def pseudo_e():
	"""
	this function iterates over all words in the training, and for each one checks if it is a low freq word.
	if it is, it changes the word to the corresponding pseudo word.
	:return: returns a dictionary that represents the emission function with the pseudo words
	"""
	pseudo_pair_count = {}
	# a dict of shape {(word / pseudo-word, tag): # of appearances}
	train_words_count = {}
	# a dict of shape {(word / pseudo-word): # of appearances}
	for word, tag in word_tag_count.keys():
		# check if word is a low frequency word
		old_word = word
		if is_low_freq(word):
			# if so, change it to pseudo word
			word = which_pseudo(word)
		# either way, if word was changed to pseudo or not,add it to the dict
		if word in train_words_count:
			train_words_count[word] += word_count[old_word]
		else:
			train_words_count[word] = word_count[old_word]
		if (word, tag) in pseudo_pair_count:
			pseudo_pair_count[(word, tag)] += word_tag_count[(old_word, tag)]
			pseudo_word_count[word] += word_count[old_word]
		else:
			pseudo_pair_count[(word, tag)] = word_tag_count[(old_word, tag)]
			pseudo_word_count[word] = word_count[old_word]
	emission = {}
	for word_tag in pseudo_pair_count.keys():
		# for each (word, tag) pair, now compute probability
		emission[word_tag] = pseudo_pair_count[word_tag] / tag_count[word_tag[TAG_INDEX]]
	return emission, train_words_count


def pseudo_word_tag():
	"""
	This function implements the most likely tag baseline, as requested in 4.b.
	:return: total error (real number e s.t. 0<=e<=1), error for known words and error for unknown words
	"""
	# p_emission is a dict of shape {(word,tag): probability of tag, given word}
	return viterbi_error_rate(p_emission)


def add1_with_pseudo_e():
	p_add1_emission = {}
	for word in train_words_after_pseudo_count.keys():
		for tag in tag_count:
			add1_denominator = (len(word_count) + tag_count[tag])
			if (word, tag) not in word_tag_count:
				p_add1_emission[(word, tag)] = 1 / add1_denominator
			else:
				# if (word, tag) exists, add 1:
				p_add1_emission[(word, tag)] = (word_tag_count[(word, tag)] + 1) / add1_denominator
	return p_add1_emission


get_data()
count_appearances()
extract_unknown()
p_emission, train_words_after_pseudo_count = pseudo_e()

# Most likely tag calculation (b.ii)
total_known_error, total_unknown_error, total_error = ml_tag()
print("MLE - Known Words Error: ", total_known_error)
print("MLE - Unknown Words Error: ", total_unknown_error)
print("MLE - Total Words Error: ", total_error, "\n")

# Base Viterbi Calculation (c.iii)
_1 = e()
base_viterbi_known, base_viterbi_unknown, base_viterbi_total = viterbi_error_rate(_1)
print("Base Viterbi - Known Words Error: ", base_viterbi_known)
print("Base Viterbi - Unknown Words Error: ", base_viterbi_unknown)
print("Base Viterbi - Total Words Error: ", base_viterbi_total, "\n")

# Viterbi with Add-1 (d.ii)
_2 = add1_e()
viterbi_add1_known, viterbi_add1_unknown, viterbi_add1_total = viterbi_error_rate(_2)
print("Viterbi with Add-1 - Known Words Error: ", viterbi_add1_known)
print("Viterbi with Add-1 - Unknown Words Error: ", viterbi_add1_unknown)
print("Viterbi with Add-1 - Total Words Error: ", viterbi_add1_total, "\n")

# Viterbi with pseudo-words and MLE
viterbi_pseudo_known, viterbi_pseudo_unknown, viterbi_pseudo_total = pseudo_word_tag()
print("Viterbi with Pseudo words - Known Words Error: ", viterbi_pseudo_known)
print("Viterbi with Pseudo words - Unknown Words Error: ", viterbi_pseudo_unknown)
print("Viterbi with Pseudo words - Total Words Error: ", viterbi_pseudo_total, "\n")

# Viterbi with Add-1 and pseudo-words
add1_p_e = add1_with_pseudo_e()
viterbi_p_add1_known, viterbi_p_add1_unknown, viterbi_p_add1_total, confusion_dict = viterbi_error_rate(add1_p_e, confusion_flag=True)
print("Viterbi with Pseudo Add-1 - Known Words Error: ", viterbi_p_add1_known)
print("Viterbi with Pseudo Add-1 - Unknown Words Error: ", viterbi_p_add1_unknown)
print("Viterbi with Pseudo Add-1 - Total Words Error: ", viterbi_p_add1_total, "\n")

print_confusion(confusion_dict)
