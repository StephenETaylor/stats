#!/usr/bin/env python3
"""
    use gensim code to generate skipgram embedding, according to 
    command line arguments:
       corpus vector_width  iters
    goal is to compare various widths and iters for quality of synactic space
    using for now score on DIACR-Ita task
"""
from gensim.models import Word2Vec, KeyedVectors
import sys
import time

corpus = 'T0.txt'
vector_width = 50
iters = 5

def main():
    global corpus, vector_width, iters
    #tiny UI
    if len(sys.argv) > 1: corpus = sys.argv[1]
    if len(sys.argv) > 2: vector_width = int(sys.argv[2])
    if len(sys.argv) > 3: iters = int(sys.argv[3])

    start = time.time()

    # read corpus into sentences
    # corpus is in CONLL format, each line is either:
    #    blank -- marks end of sentence
    #    raw POS lemma

    sentences = []
    sentence = []
    with open(corpus) as fi:
        for lin in fi: 
            line = lin.strip().split()
            if len(line) != 3:  #assume end of sentence
                sentences.append(sentence)
                sentence = []
            else: sentence.append(line[2])

    now = time.time()
    print (now-start, 'sec', len(sentences), 'sentences')
    # train embedding with gensim
    model = Word2Vec(sentences, min_count=5, size=vector_width, workers=3, window=5, sg=1, iter=iters, negative=5)

    saving = time.time()

    print ('saving', saving-now, 'sec')
    # save and exit
    file_name = 'w2v.'+  corpus   +'.'+str(vector_width)+'_window-5_iter-'+str(iters) + '.bin'
    model.wv.save(file_name)

    print (file_name)
    

main()
