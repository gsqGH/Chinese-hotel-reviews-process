#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
import gensim
import numpy as np
import pickle as pkl
from embeddings.random_vec import RandomVec

CWD = os.path.split(os.path.realpath(__file__))[0]
WORD_DIM = 300
model_file = os.path.join(CWD, '../data/pickles/GoogleNews-vectors-negative300.bin')
glove_file = os.path.join(CWD, '../data/pickles/glove.6B.300d.txt')
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
# model = gensim.models.KeyedVectors.load_word2vec_format(glove_file,binary=False)
rVec = RandomVec(WORD_DIM)

def findMaxLength(fileName):
    with open(fileName, 'r') as f:
        temp = 0
        max_length = 0
        count = {}
        for line in f:
            if line in ['\n', '\r\n']:
                if temp > max_length:
                    max_length = temp
                if temp in count :
                    count[temp] += 1
                else :
                    count[temp] = 1
                temp = 0
            else:
                temp += 1

    return max_length, count

def get_embedding(fileName, Vec, Tag, label, structure):
    word = []
    tag = []

    sentence = []
    sentence_tag = []
    sentence_label = []
    sentence_structure = []

    with open(fileName, 'r') as f:
        l = np.array([0,0])
        location = -1
        for line in f:
            if line in ['\n', '\r\n']:
                sentence.append(word)
                sentence_tag.append(np.array(tag))
                sentence_label.append(l)
                sentence_structure.append(location)
                l = np.array([0,0])
                location = -1
                word = []
                tag = []
            else:
                assert(len(line.split()) == 4)
                w = line.split()[0].lower()
                try:
                    temp = model[w]
                except:
                    temp = rVec[w]
                word.append(temp)

                argue = line.split()[2]
                if argue == 'True':
                    l = np.array([1,0])
                elif argue == 'False':
                    l = np.array([0,1])

                location = line.split()[1]

                t = line.split()[3]
                # 3classes 0-O, 1-B, 2-I
                if t == 'O':
                    tag.append(np.array([1,0,0]))
                elif t == 'B':
                    tag.append(np.array([0,1,0]))
                elif t == 'I':
                    tag.append(np.array([0,0,1]))
                else:
                    print('error in input label')
                    sys.exit(0)

        assert(len(sentence) == len(sentence_tag))
        # print(len(sentence[0]))
        # print(len(sentence[0][0]))
        # print(sentence_tag[0])
        # print(len(sentence_label))
        # print(len(sentence_structure))
        # print(sentence_structure[:10])
        pkl.dump(sentence, open(Vec, 'wb'))
        pkl.dump(sentence_tag, open(Tag, 'wb'))
        pkl.dump(np.array(sentence_label), open(label, 'wb'))
        pkl.dump(np.array(sentence_structure), open(structure, 'wb'))

if __name__ == '__main__':
    train_file = os.path.join(CWD, '../data/train.data')
    train_vec = os.path.join(CWD, '../data/pickles/train_word_vec_' + str(WORD_DIM))
    train_tag = os.path.join(CWD, '../data/pickles/train_tag')
    train_label = os.path.join(CWD, '../data/pickles/train_label')
    train_structure = os.path.join(CWD, '../data/pickles/train_structure')

    test_file = os.path.join(CWD, '../data/test.data')
    test_vec = os.path.join(CWD, '../data/pickles/test_word_vec_' + str(WORD_DIM))
    test_tag = os.path.join(CWD, '../data/pickles/test_tag')
    test_label = os.path.join(CWD, '../data/pickles/test_label')
    test_structure = os.path.join(CWD, '../data/pickles/test_structure')

    maxlen = 73
    # maxlen = max(findMaxLength(train_file)[0], findMaxLength(test_file)[0])

    get_embedding(train_file, train_vec, train_tag, train_label, train_structure)
    get_embedding(test_file, test_vec, test_tag, test_label, test_structure)
    print('transported training and testing data into word vectors and saved in pickles')
