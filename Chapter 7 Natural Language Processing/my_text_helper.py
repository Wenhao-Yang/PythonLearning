#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: my_text_helper.py
@Time: 2019/3/9 22:34
@Overview:
"""
#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: text_helpers.py
@Time: 2019/3/9 20:12
@Overview: The major function of producing data
"""
import os
import string
import tarfile
import collections
import numpy as np
import requests


def load_movie_data():
    save_folder_name = '../LocalData'
    save_file_name = os.path.join(save_folder_name, 'rt-polaritydata.tar.gz')
    pos_file = os.path.join(save_folder_name, 'rt-polaritydata','rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polaritydata','rt-polarity.neg')

    # if not os.path.exists(save_file_name):
    #     movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    #     stream_data = urllib.request.urlopen(movie_data_url)
    #     with open(save_file_name, 'wb') as f:
    #         f.write(stream_data.read())

    if not os.path.exists(os.path.join(save_folder_name, 'rt-polaritydata')):
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

        # Save tar.gz file
        req = requests.get(movie_data_url, stream=True)
        with open(save_file_name, 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
    if not (os.path.exists(pos_file) and os.path.exists(neg_file)):
        # tar_file = tarfile.open('../LocalData/rt-polaritydata.tar.gz', mode='r:gz')
        # pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
        # neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')
        # pos_data = []
        # for line in pos:
        #         pos_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
        # neg_data = []
        # for line in neg:
        #         neg_data.append(line.decode('ISO-8859-1').encode('ascii', errors='ignore').decode())
        # tar_file.close()
        #
        # with open(pos_file, 'w') as pos_file_handler:
        #         pos_file_handler.write(''.join(pos_data))
        # with open(neg_file, 'w') as neg_file_handler:
        #         neg_file_handler.write(''.join(neg_data))
        # Extract tar.gz file into temp folder
        tar = tarfile.open(save_file_name, "r:gz")
        tar.extractall(path=save_folder_name)
        tar.close()

    pos_data = []
    neg_data = []
    with open(pos_file, 'r', encoding='latin-1') as temp_pos_file:
        for row in temp_pos_file:
            pos_data.append(row.encode('ascii', errors='ignore').decode())
    temp_pos_file.close()
    with open(neg_file, 'r', encoding='latin-1') as temp_neg_file:
        for row in temp_neg_file:
            neg_data.append(row.encode('ascii', errors='ignore').decode())
    temp_neg_file.close()
    neg_data = [x.rstrip() for x in neg_data]
    texts = pos_data + neg_data
    target = [1]*len(pos_data) + [0]*len(neg_data)
    return(texts, target)

def normalize_text(texts, stops):
    texts = [x.lower() for x in texts]
    # Remove punctuation
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # Remove numbers
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    # Remove stopwords
    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]
    # Trim extra whitespace
    texts = [' '.join(x.split()) for x in texts]
    return (texts)

# Define a function that creates a dictionary of words with their count
def build_dictionary(sentences, vocabulary_size):
    # Turn sentences (list of strings) into lists of words
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist]
    # Initialize list of [word, word_count] for each word, starting with unknown
    count = [['RARE', 1]]
    # Ass most frequent words, limited to the N-most frequent
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    word_dict = {}
    # For each word, that we want in the dictionary, add it, then make it the value of the prior dictionary length
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    return (word_dict)

# Declare the function that will convert a list of sentences into lists of word indices that we can pass into our embedding lookup function.
def text_to_numbers(sentences, word_dict):
    data = []
    for sentence in sentences:
        sentence_data = []
        # For each word, either use selected index or rare word index
        for word in sentence:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return (data)

# Define a function that will return the skip-gram batches. Train on pairs of words where one word is the training input(from the target word at the center of the window) and the other word is selected from the window.
def generate_batch_data(sentences, batch_size, window_size, method='skip-gram'):
    # Full up data batch
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        # Select random sentence to start
        rand_sentence = np.random.choice(sentences)
        # Generate consecutive windows to look at
        window_sequences = [rand_sentence[max((ix-window_size), 0):(ix+window_size+1)] for ix, x in enumerate(rand_sentence)]
        # Denote which element of each window is the center word of interest
        label_indices = [ix if ix<window_size else window_size for ix,x in enumerate(window_sequences)]
        # Pull out center word of interest for each window and create a tuple for each window
        batch, labels = [], []
        if method=='skip-gram':
            batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x, y in zip(window_sequences, label_indices)]
            # Make it into a big list of tuples(target word, surrounding word)
            tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
            if len(tuple_data) > 0:
                batch, labels = [list(x) for x in zip(*tuple_data)]
        elif method=='cbow':
            batch_and_labels = [(x[:y] + x[(y+1):], x[y]) for x,y in zip(window_sequences, label_indices)]
            # ONly keep windows with consistent 2*window size
            batch_and_labels = [(x, y) for x,y in batch_and_labels if len(x)==2*window_size]
            if len(batch_and_labels) > 0:
                batch, labels = [list(x) for x in zip(*batch_and_labels)]
        else:
            raise ValueError('Method {} not implementd yet.'.format(method))

        # Extract batch and label
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])

    # Trim batch and label at the enc
    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]

    # Convet to numpy array
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return batch_data, label_data
