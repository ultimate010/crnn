import cPickle
import numpy as np
from process_sst_data import SentimentPhrase

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5, pad_left=True):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    if pad_left:
        for i in xrange(pad):
            x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def get_label(sentiment):
    if sentiment >= 0 and sentiment <= 0.2:
        return 0
    elif sentiment > 0.2 and sentiment <= 0.4:
        return 1
    elif sentiment > 0.4 and sentiment <= 0.6:
        return 2
    elif sentiment > 0.6 and sentiment <= 0.8:
        return 3
    elif sentiment > 0.8 and sentiment <= 1.0:
        return 4
    else:
        return -1


def make_idx_data_cv(phrases, sentences, word_idx_map, max_l=51, k=300, filter_h=5, pad_left=True):
    """
    Transforms sentences into a 2-d matrix.
    """
    debug = True
    if debug:
        train_file = open('sst1_train.txt', 'w')
        test_file = open('sst1_test.txt', 'w')
    train, test = [], []
    for p in phrases:  # put all phrase into train data
        sent = get_idx_from_sent(' '.join(p.words), word_idx_map, max_l, k, filter_h, pad_left=pad_left)
        sent.append(get_label(p.sentiment))
        if debug:
            train_file.write('%s %d\n' % (' '.join(p.words), get_label(p.sentiment)))
        train.append(sent)

    for s in sentences:
        sent = get_idx_from_sent(' '.join(s.words), word_idx_map, max_l, k, filter_h, pad_left=pad_left)
        sent.append(get_label(s.sentiment))
        if s.split == 'train':
            train.append(sent)
            if debug:
                train_file.write('%s %d\n' % (' '.join(s.words), get_label(s.sentiment)))
        elif s.split == 'dev':
            train.append(sent)
            if debug:
                train_file.write('%s %d\n' % (' '.join(s.words), get_label(s.sentiment)))
        else:
            test.append(sent)
            if debug:
                test_file.write('%s %d\n' % (' '.join(s.words), get_label(s.sentiment)))
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]

x = None
def load_data(pad_left=True):
    global x
    if x is None:
        x = cPickle.load(open("sst1.p","rb"))
    phrases, sentences, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4], x[5]
    datasets = make_idx_data_cv(phrases, sentences, word_idx_map, max_l=56, k=300, filter_h=5, pad_left=pad_left)
    img_h = len(datasets[0][0])-1
    return datasets[0][:,:img_h], datasets[0][:, -1], datasets[1][:,: img_h], datasets[1][: , -1], W, W2

if __name__ == '__main__':
    load_data()
