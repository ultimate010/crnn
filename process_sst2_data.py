#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Code from  https://github.com/gojomo/gensim/blob/f5b4e30dbc3ea71520b61e9d586cc946e2388ae0/gensim/test/test_doc2vec.py#L282
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import logging
from collections import namedtuple, defaultdict
from six.moves import zip as izip
import cPickle
import sys
import os
import numpy as np
import pandas as pd

class SentimentPhrase(object):
    def __init__(self, words, tags, split, sentiment, sentence_id):
        self.words = words
        self.tags = tags
        self.split = split
        self.sentiment = sentiment
        self.sentence_id = sentence_id

    def __str__(self):
        return '%s %s %s %s %s' % (self.words, self.tags, self.split, self.sentiment, self.sentence_id)

def read_su_sentiment_rotten_tomatoes(dirname, lowercase=True):
    """
    Read and return documents from the Stanford Sentiment Treebank
    corpus (Rotten Tomatoes reviews), from http://nlp.Stanford.edu/sentiment/
    Initialize the corpus from a given directory, where
    http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
    has been expanded. It's not too big, so compose entirely into memory.
    """
    logging.info("loading corpus from %s" % dirname)

    # many mangled chars in sentences (datasetSentences.txt)
    chars_sst_mangled = ['à', 'á', 'â', 'ã', 'æ', 'ç', 'è', 'é', 'í',
                         'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'û', 'ü']
    sentence_fixups = [(char.decode('utf-8').encode('latin1'), char) for char in chars_sst_mangled]
    # more junk, and the replace necessary for sentence-phrase consistency
    sentence_fixups.extend([
        ('Â', ''),
        ('\xa0', ' '),
        ('-LRB-', '('),
        ('-RRB-', ')'),
    ])
    # only this junk in phrases (dictionary.txt)
    phrase_fixups = [('\xa0', ' ')]

    # sentence_id and split are only positive for the full sentences

    # read sentences to temp {sentence -> (id,split) dict, to correlate with dictionary.txt
    vocab = defaultdict(float)
    info_by_sentence = {}
    with open(os.path.join(dirname, 'datasetSentences.txt'), 'r') as sentences:
        with open(os.path.join(dirname, 'datasetSplit.txt'), 'r') as splits:
            next(sentences)  # legend
            next(splits)     # legend
            for sentence_line, split_line in izip(sentences, splits):
                (id, text) = sentence_line.split('\t')
                id = int(id)
                text = text.rstrip()
                for junk, fix in sentence_fixups:
                    text = text.replace(junk, fix)
                (id2, split_i) = split_line.split(',')
                assert id == int(id2)
                if text not in info_by_sentence:    # discard duplicates
                    info_by_sentence[text] = (id, int(split_i))
                else:
                    logging.info('Duplicates: %s' % text)

    # read all phrase text
    phrases = [None] * 239232  # known size of phrases
    with open(os.path.join(dirname, 'dictionary.txt'), 'r') as phrase_lines:
        for line in phrase_lines:
            (text, id) = line.split('|')
            for junk, fix in phrase_fixups:
                text = text.replace(junk, fix)
            phrases[int(id)] = text.rstrip()  # for 1st pass just string

    test_str = ''
    train_str = ''
    with open(os.path.join(dirname, 'sentiment_labels.txt'), 'r') as sentiments:
        next(sentiments)  # legend
        for line in sentiments:
            (id, sentiment) = line.split('|')
            id = int(id)
            sentiment = float(sentiment)
            text = phrases[id]
            (sentence_id, split_i) = info_by_sentence.get(text, (None, 0))
            if split_i == 2:  # test data
                test_str += text
            elif split_i == 1:
                train_str += text

    # add sentiment labels, correlate with sentences
    with open(os.path.join(dirname, 'sentiment_labels.txt'), 'r') as sentiments:
        next(sentiments)  # legend
        for line in sentiments:
            (id, sentiment) = line.split('|')
            id = int(id)
            sentiment = float(sentiment)
            text = phrases[id]
            words = text.split()
            if lowercase:
                words = [word.lower() for word in words]
            (sentence_id, split_i) = info_by_sentence.get(text, (None, 0))
            if sentence_id is not None:
                for w in words:
                    vocab[w]+=1
            split = [None, 'train', 'test', 'dev'][split_i]
            if sentence_id is None and (text in test_str or text not in train_str):  # skip phrase in test sentences and no substr of train sentences
                phrases[id] = SentimentPhrase(words, [id], split, 0.5, sentence_id)  # 0.5 for remove
            else:
                phrases[id] = SentimentPhrase(words, [id], split, sentiment, sentence_id)

    logging.info("loaded corpus with %i sentences and %i phrases from %s",
                 len(info_by_sentence), len(phrases), dirname)

    # counts don't match 8544, 2210, 1101 because 13 TRAIN and 1 DEV sentences are duplicates
    # print len([phrase for phrase in phrases if phrase.split == 'train']) # == 8531  # 'train'
    # print len([phrase for phrase in phrases if phrase.split == 'test']) # == 2210  # 'test'
    # print len([phrase for phrase in phrases if phrase.split == 'dev']) # == 1100  # 'dev'
    phrase0 = [phrase for phrase in phrases if phrase.sentence_id is None
              and (phrase.sentiment > 0.6 or phrase.sentiment <= 0.4) ]
    sentences = [phrase for phrase in phrases if phrase.sentence_id is not None
              and (phrase.sentiment > 0.6 or phrase.sentiment <= 0.4) ]
    print 'sentences %d phrase %d vocab %d' % (len(sentences),
                                               len(phrase0), len(vocab))
    print 'Data example'
    print phrase0[10]
    print sentences[10]
    return phrase0, sentences, vocab

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

if __name__ == '__main__':
    w2v_file = sys.argv[1] # w2v pretrain file
    sst_folder = sys.argv[2] # sst folder
    print 'load corpus'
    phrases, sentences, vocab = read_su_sentiment_rotten_tomatoes(sst_folder)
    print 'load word2vec'
    max_l = np.max([len(s.words) for s in sentences])
    print "number of sentences: " + str(len(sentences))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    w2v = load_bin_vec(w2v_file, vocab)
    print "%d words in w2v" % len(w2v)
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([ phrases, sentences, W, W2, word_idx_map, vocab], open("sst2.p", "wb"))
    x = cPickle.load(open("sst2.p","rb"))
    phrases, sentences, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4], x[5]
    print "dataset created!"
