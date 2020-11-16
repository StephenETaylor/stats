#!/usr/bin/env python3
"""
    using args on command line: dim_size  iters  
    and the file target.txt,
     0. find appropriate embeddings in unzipped_embeddings for T0.txt and T1.txt
     1. create a cross-lingual transformation
     2. compare each target word
     3. create a threshold
     4. assign binary and rank values
     5. output results into an appropriate results directory
"""
from gensim.models import Word2Vec

import math
import os
import sys
import time

#import ccaxform1
import compare2

iters = 10
emb_dim = 100
emb_dir = 'unzipped_embeddings/'
compare2.general_folder = 'post-test-29-a'# this should contain some clue about embedding

compare2.reverse_emb = True
compare2.use_nearest_neigbh = False
compare2.use_bin_thld = True
compare2.emb_type = 'w2v'
compare2.emb_dim = 275
compare2.window = 5 #10 #5
compare2.iter = 10
compare2.ver_xform = 'python'
compare2.individual_xform = False
compare2.compute_trans_qual = True
compare2.normalization = 0
compare2. thirdspace = False
compare2.transdict_path = 'data/tmp/trans.dict'
compare2.transdict_base = 1000
compare2.transdict_length = 5000
compare2.transdict_limit = 25000
compare2.compare_gold = True


def main():
    global iters, emb_dim,  task_1_dir, task_2_dir, folder_to_zip
    global zip_file
    trial = ''
    if len(sys.argv) > 1: emb_dim = int(sys.argv[1])
    if len(sys.argv) > 2: iters = int(sys.argv[2])
    if len(sys.argv) > 3: trial = int(sys.argv[3])
    
    # initialize trial folder name
    compare2.general_folder = (str(emb_dim) + '-' + str(iters) + '-' + str(trial) + 
                        '+' + str(compare2.transdict_length))
    task_1_dir, task_2_dir, folder_to_zip, zip_file = compare2.init_folders(compare2.general_folder)

    save_file = os.path.join(task_2_dir, 'italian.txt')
    save_file_binary = os.path.join(task_1_dir, 'italian.txt')

    model1path = get_model('T0.txt', emb_dim, iters)
    model2path = get_model('T1.txt', emb_dim, iters)
    #with open('targets.txt') as fi:
    #    targets = fi.read().split()
    #    print (targets)
    rho, acc, bin_thld, min_neighb_cnt = compare2.compare(
                            model1path, model2path, 'targets.txt' ,
                            'gold1.txt', 'gold2.txt', # gold files
                            False, True, False, #use_bin_thld
                            save_file_ranks = save_file,
                            save_file_binary = save_file_binary,
                            one_minus = True)
    print(rho,acc,bin_thld, min_neighb_cnt)



def get_model(fn, emb_dim, iters):
    """
        fn is the name of the corpus on which the embedding is based
        emb_dim is the width of the embedding
        iters is the number of epochs it was trained for
    """

    corp_emb_path =  (emb_dir + 'w2v.' + fn + '.' + str(emb_dim) + 
                        '_window-5_iter-' + str(iters) + '.bin')
    #model = Word2Vec.load(corp_emb_path)
    #return model
    return corp_emb_path


if __name__ == '__main__': main()
