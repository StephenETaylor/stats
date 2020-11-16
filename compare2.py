
import glob
import shutil
import sys
import os.path as op
import scipy.stats as ss
import os
import pathlib
import shutil
import numpy as np
import itertools as it

from gensim.models import KeyedVectors
import gensim.matutils as gma

from config import EMBEDDINGS_EXPORT_PATH, DATA_DIR, TMP_DIR, ENGLISH_TEST_TARGET_WORDS, TEST_DATA_RESULTS_DIR, \
    GERMAN_TEST_TARGET_WORDS, LATIN_TEST_TARGET_WORDS, SWEDISH_TEST_TARGET_WORDS, TEST_DATA_TRUTH_ANSWER_TASK_1, \
    TEST_DATA_TRUTH_ANSWER_TASK_2, SWEDISH_TEST_GOLD_TASK_1, SWEDISH_TEST_GOLD_TASK_2, LATIN_TEST_GOLD_TASK_1, \
    LATIN_TEST_GOLD_TASK_2, GERMAN_TEST_GOLD_TASK_1, GERMAN_TEST_GOLD_TASK_2, ENGLISH_TEST_GOLD_TASK_2, \
    ENGLISH_TEST_GOLD_TASK_1
from data.post_eval_data.scoring_program.evaluation_official import spearman_official, accuracy_official
from embeddings.sense_comparator import load_transform_matrix, compare_sense
import random


sys.path.insert(0,os.getcwd())
print(sys.path)
print (os.getcwd(),flush=True)

ver_xform = None   # values:  'java', 'python'; which xform builder to use
individual_xform = None # values True, False.  custom xform foreach target word
transdict_length = None # if not None, limit number of translation dict entries
compute_trans_qual = None
normalization = None
thirdspace = None
transdict_path = None
transdict_base = None
transdict_limit = None
general_folder = None
compare_gold = True

def main():
    global ver_xform, individual_xform, transdict_length, compute_trans_qual, normalization, thirdspace, transdict_path, transdict_base, transdict_limit, general_folder, compare_gold
    # clmet_t1 = os.path.join(EMBEDDINGS_EXPORT_PATH, 'fasttext-min-count-5', 'clmet_t1','lower_fasttext.clmet_t1.100_window-5_iter-10.vec')
    # clmet_t2 = os.path.join(EMBEDDINGS_EXPORT_PATH, 'fasttext-min-count-5', 'clmet_t2','lower_fasttext.clmet_t2.100_window-5_iter-10.vec')
    #
    # target_words = os.path.join(DATA_DIR,'sense_words','target_words_labeled')

    # compare(clmet_t1, clmet_t2, target_words)

    # default pro binar vezmu druhy nejvetsi velikost pruniku a tu vydelim dvema

    # run_durel()



    general_folder = 'post-test-29-a'
    #task_1_dir, task_2_dir, folder_to_zip, zip_file = init_folders(general_folder)

    reverse_emb = True
    use_nearest_neigbh = False
    use_bin_thld = True
    emb_type = 'w2v'
    emb_dim = 275
    window = 10 #5
    iter = 10
    ver_xform = 'python'
    individual_xform = False
    compute_trans_qual = True
    normalization = 0
    thirdspace = False
    transdict_path = 'data/tmp/trans.dict'
    transdict_base = 1000
    compare_gold = True
    
    # little UI to alter these for graphs
    state = 0
    for x in sys.argv[1:]:
        if state == 0 and x == '-reverse':
            reverse_emb = True
        elif state == 0 and x == '-noreverse':
            reverse_emb = False
        elif state == 0 and x == '-qualxform':
            compute_trans_qual = True
        elif state == 0 and x == '-noqualxform':
            compute_trans_qual = False
        elif state == 0 and x == '-thirdspace':
            thirdspace = True
        elif state == 0 and x == '-nothirdspace':
            thirdspace = False
        elif state == 0 and x == '-binaryThreshold':
            use_nearest_neigbh = False
            use_bin_thld = True
        elif state == 0 and x == '-nobinaryThreshold':
            use_nearest_neigbh = True
            use_bin_thld = False
        elif state == 0 and x == '-xformpython':
            ver_xform = 'python'
        elif state == 0 and x == '-xformjava':
            ver_xform = 'java'
        elif state == 0 and x == '-noeachtargetxform':
            individual_xform = False
        elif state == 0 and x == '-eachtargetxform':
            individual_xform = True
        elif state == 0 and x == '-nocomparegold':
            compare_gold = False
        elif state == 0 and x == '-comparegold':
            compare_gold = True
        elif state == 0 and x == '-window:':
            state = 2
        elif state == 2:
            window = int(x)
            state = 0
        elif state == 0 and x == '-embeddingSize:':
            state = 1
        elif state == 1:
            emb_dim = int(x)
            state = 0
        elif state == 0 and x == '-iter:':
            state = 3
        elif state == 3:
            iter = int(x)
            state = 0
        elif state == 0 and x == '-transdict_length:':
            state = 4
        elif state == 4:
            transdict_length = int(x)
            state = 0
        elif state == 0 and x == '-normalize:':
            state = 5
        elif state == 5:
            normalization = 0
            for letter in x:
                if letter == '0': next
                elif letter == 'c' : normalization = normalization | 1
                elif letter == 'u' : normalization = normalization | 2
                else: cry('only 0, c, cu anticipated normalizations')
            state = 0
        elif state == 0 and x == '-transdict:':
            state = 6
        elif state == 6:
            transdict_path = x
            state = 0
        elif state == 0 and x == '-transdict_base:':
            state = 7
        elif state == 7:
            transdict_base = int(x)
            state = 0
        elif state == 0 and x == '-general_folder:':
            state = 8
        elif state == 8:
            general_folder = x
            state = 0
        elif state == 0 and x == '-transdict_limit:':
            state = 9
        elif state == 9:
            transdict_limit = int(x)
            state = 0
        else:
            print(x)
            cry('non-existent command-line flag') # terminate with traceback

    if transdict_length != None and transdict_limit != None and  transdict_base != None and transdict_limit - transdict_base <= transdict_length:
            cry('contradictory limit, base, length')


    task_1_dir, task_2_dir, folder_to_zip, zip_file = init_folders(general_folder)
    acc_list = []
    rho_list = []

    # #
    acc, rho = run_english_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter)
    acc_list.append(acc)
    rho_list.append(rho)

    acc, rho = run_german_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter)
    acc_list.append(acc)
    rho_list.append(rho)

    acc, rho = run_latin_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter)
    acc_list.append(acc)
    rho_list.append(rho)

    acc, rho = run_swedish_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh, emb_type, emb_dim, window, iter)
    acc_list.append(acc)
    rho_list.append(rho)

    acc_avg = round(np.mean(acc_list), 3)
    rho_avg = round(np.mean(rho_list), 3)

    print('Type' + '\t' + 'avg acc/rank' + '\t' + 'english' + '\t' + 'german' + '\t' + 'latin'+ '\t' + 'swedish' + '\t' + 'reverse emb'
          + '\t' + 'emb_type' + '\t' + 'emb_dim' + '\t' + 'window' + '\t' + 'iter' + '\t' + 'use bin thld' + '\t' + 'use nearest neigh')
    print("Binary overview" + '\t' + str(acc_avg) +
          '\t' + str(acc_list[0]) + '\t' + str(acc_list[1]) + '\t' + str(acc_list[2]) + '\t' + str(acc_list[3]) + '\t'
          + str(reverse_emb) + '\t' + emb_type + '\t' + str(emb_dim) + '\t' + str(window) + '\t' + str(iter)
          + '\t' + str(use_bin_thld) + '\t' + str(use_nearest_neigbh))


    print('Rank overview' + '\t' + str(rho_avg) +
          '\t' + str(rho_list[0]) + '\t' + str(rho_list[1]) + '\t' + str(rho_list[2]) + '\t' + str(rho_list[3]) + '\t'
          + str(reverse_emb) + '\t' + emb_type + '\t' + str(emb_dim) + '\t' + str(window) + '\t' + str(iter))



    # print('Type' + '\t' + 'avg acc/rank' + '\t' + 'english' + '\t' + 'german' + '\t' + 'latin'+ '\t' + 'swedish' + '\t' + 'reverse emb'
    #       + '\t' + 'emb_type' + '\t' + 'emb_dim' + '\t' + 'window' + '\t' + 'iter' + '\t' + 'use bin thld' + '\t' + 'use nearest neigh')
    # print("Binary overview" + '\t' + str(acc_avg) +
    #       '\t' + str(acc_list[0]) + '\t' + str(0.0) + '\t' + str(acc_list[1]) + '\t' + str(acc_list[2]) + '\t'
    #       + str(reverse_emb) + '\t' + emb_type + '\t' + str(emb_dim) + '\t' + str(window) + '\t' + str(iter)
    #       + '\t' + str(use_bin_thld) + '\t' + str(use_nearest_neigbh))
    #
    #
    # print('Rank overview' + '\t' + str(rho_avg) +
    #       '\t' + str(rho_list[0]) + '\t' + str(0.0) + '\t' + str(rho_list[1]) + '\t' + str(rho_list[2]) + '\t'
    #       + str(reverse_emb) + '\t' + emb_type + '\t' + str(emb_dim) + '\t' + str(window) + '\t' + str(iter))


    # #
    # # zip_folder(folder_to_zip, zip_file)
    #


    # compute_spearman_between_res()
    # evaluate_submission_results()
    pass


def evaluate_submission_results():
    submissions = ['default', 'default_binary_threshold', 'default_reveresed_binary_threshold', 'default_reversed',
                   'LDA-100', 'LDA-100-globalThreshold', 'map-ort-i', 'map-ort-i-globalThreshold', 'map-unsup', 'map-unsup-globalThreshold']
    languages = ['english', 'german', 'latin', 'swedish']
    for sub in submissions:
        print('-' * 70)
        print('-' * 70)
        print('-' * 70)
        print('Evalaluating submission named:', sub)
        for lang in languages:
            print('-' * 50)
            print('Lang:' + lang)
            binary_gold_file = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_1, lang + '.txt')
            binary_pred_file = os.path.join(TEST_DATA_RESULTS_DIR, sub, 'answer','task1', lang + '.txt')

            rank_gold_file = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_2, lang + '.txt')
            rank_pred_file = os.path.join(TEST_DATA_RESULTS_DIR, sub, 'answer', 'task2', lang + '.txt')
            my_rho, my_pval = compute_spearman(rank_gold_file, rank_pred_file, print_res=False)
            print('My results: Rho:' + str(my_rho) + ' p-value:' + str(my_pval))

            off_rho, off_pval = spearman_official(rank_gold_file, rank_pred_file)
            print('Official results: Rho:' + str(off_rho) + ' p-value:' + str(off_pval))

            acc_official = accuracy_official(binary_gold_file, binary_pred_file)
            print('Official accuracy:' + str(acc_official))



# todo udelat ukladani do pozadovaneho formatu, moznost pridat  tam  k tomu poznamku

# todo pridat parametry do metod aby se to generovalo automaticky embeddingy podle vstupu
def run_swedish_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
                        emb_type, emb_dim, window, iter):
    print('-' * 70)
    print('Swedish')
    save_file = os.path.join(task_2_dir, 'swedish.txt')
    save_file_binary = os.path.join(task_1_dir, 'swedish.txt')

    # config
    # corp1_emb_file = 'w2v.swedish_corpus1.100_window-5_iter-5_min-count-5.vec'
    # corp2_emb_file = 'w2v.swedish_corpus2.100_window-5_iter-5_min-count-5.vec'
    corp1_emb_file = emb_type + '.swedish_corpus1.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'
    corp2_emb_file = emb_type + '.swedish_corpus2.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'


    corp_1_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'swedish_corpus_1', corp1_emb_file)
    corp_2_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'swedish_corpus_2', corp2_emb_file)

    target_words = SWEDISH_TEST_TARGET_WORDS
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, SWEDISH_TEST_GOLD_TASK_1, SWEDISH_TEST_GOLD_TASK_2,
                               reverse_emb, use_bin_thld, use_nearest_neigbh, save_file_ranks=save_file, save_file_binary=save_file_binary,
                               one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'swedish' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(min_neighb_cnt) +'\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho

def run_latin_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
                      emb_type, emb_dim, window, iter):
    print('-' * 70)
    print('Latin')
    save_file = os.path.join(task_2_dir, 'latin.txt')
    save_file_binary = os.path.join(task_1_dir, 'latin.txt')

    # corp1_emb_file = 'w2v.latin_corpus1.100_window-5_iter-5_min-count-5.vec'
    # corp2_emb_file = 'w2v.latin_corpus2.100_window-5_iter-5_min-count-5.vec'
    corp1_emb_file = emb_type + '.latin_corpus1.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'
    corp2_emb_file = emb_type + '.latin_corpus2.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'

    corp_1_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'latin_corpus_1', corp1_emb_file)
    corp_2_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'latin_corpus_2', corp2_emb_file)

    target_words = LATIN_TEST_TARGET_WORDS
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, LATIN_TEST_GOLD_TASK_1, LATIN_TEST_GOLD_TASK_2,
                               reverse_emb, use_bin_thld, use_nearest_neigbh, save_file_ranks=save_file, save_file_binary=save_file_binary,
                               one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'latin' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(
        min_neighb_cnt) + '\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho


def run_german_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
                       emb_type, emb_dim, window, iter):
    print('-' * 70)
    print('German')
    save_file = os.path.join(task_2_dir, 'german.txt')
    save_file_binary = os.path.join(task_1_dir, 'german.txt')

    # corp1_emb_file = 'w2v.german_corpus1.100_window-5_iter-5_min-count-5.vec'
    # corp2_emb_file = 'w2v.german_corpus2.100_window-5_iter-5_min-count-5.vec'
    corp1_emb_file = emb_type + '.german_corpus1.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'
    corp2_emb_file = emb_type + '.german_corpus2.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(
        iter) + '_min-count-5.vec'

    corp_1_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'german_corpus_1', corp1_emb_file)
    corp_2_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'german_corpus_2', corp2_emb_file)

    target_words = GERMAN_TEST_TARGET_WORDS
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, GERMAN_TEST_GOLD_TASK_1, GERMAN_TEST_GOLD_TASK_2,
                                 reverse_emb, use_bin_thld, use_nearest_neigbh, save_file_ranks=save_file, save_file_binary=save_file_binary,
                                 one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'german' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(
        min_neighb_cnt) + '\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho


def run_english_default(task_1_dir, task_2_dir, reverse_emb, use_bin_thld, use_nearest_neigbh,
                        emb_type, emb_dim, window, iter):
    print('-' * 70)
    print('English')
    save_file = os.path.join(task_2_dir, 'english.txt')
    save_file_binary = os.path.join(task_1_dir, 'english.txt')

    # corp1_emb_file = 'w2v.english_corpus1.100_window-5_iter-5_min-count-5.vec'
    # corp2_emb_file = 'w2v.english_corpus2.100_window-5_iter-5_min-count-5.vec'
    corp1_emb_file = emb_type + '.english_corpus1.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(iter) + '_min-count-5.vec'
    corp2_emb_file = emb_type + '.english_corpus2.' + str(emb_dim) + '_window-' + str(window) + '_iter-' + str(iter) + '_min-count-5.vec'


    corp_1_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'english_corpus_1', corp1_emb_file)
    corp_2_emb = os.path.join(EMBEDDINGS_EXPORT_PATH, 'english_corpus_2', corp2_emb_file)

    target_words = ENGLISH_TEST_TARGET_WORDS
    rho, acc, bin_thld, min_neighb_cnt = compare(corp_1_emb, corp_2_emb, target_words, ENGLISH_TEST_GOLD_TASK_1, ENGLISH_TEST_GOLD_TASK_2,
                       reverse_emb, use_bin_thld, use_nearest_neigbh, save_file_ranks=save_file, save_file_binary=save_file_binary,
                        one_minus=True)

    print('Config:')
    print(str(reverse_emb) + '\t' + 'english' + '\t' + corp1_emb_file + '\t' + corp2_emb_file + '\t' + str(
        min_neighb_cnt) + '\t' + str(bin_thld) +
          '\t' + str(acc) + '\t' + str(rho))

    return acc, rho

def run_durel():
    durel18 = os.path.join(EMBEDDINGS_EXPORT_PATH, 'durel', 'dta18.txt.w2v.vec')
    durel19 = os.path.join(EMBEDDINGS_EXPORT_PATH, 'durel', 'dta19.txt.w2v.vec')

    target_words = os.path.join(DATA_DIR, 'sense_words', 'durel_wl_dta.ranks')
    # xform = os.path.join(DATA_DIR, 'xform-mat', 'dta18.txt-dta19.txt.xform')
    xform = os.path.join(DATA_DIR, 'xform-mat', 'dta18.txt-dta19.txt.xform')
    # compare(durel19, durel18, target_words, xform=None, run_spearman=True)


def compare(src_emb_path, trg_emb_path, target_words_path, gold_file_task1, gold_file_task2, reverse, use_binary_threshold,
            use_nearest_neigbhrs,
            xform=None, max_links=100000, run_spearman=None, save_file_ranks=None, save_file_binary=None,
            one_minus=False, topn=100):
    # allow overriding gold evaluation (e.g. if gold doesn't exist, or is junk)
    if run_spearman == None:
        if compare_gold == None: run_spearman = True
        else: run_spearman = compare_gold
    # and otherwise, run_spearman was specified in the call; go with that value

    # delete_tmp_dir()

    # reversing
    if reverse is True:
        tmp_path = src_emb_path
        src_emb_path = trg_emb_path
        trg_emb_path = tmp_path


    print("Running comparison for with topn:" + str(topn) + " min_neighbours_count:" + str(use_nearest_neigbhrs) +" use binary threshold:" + str(use_binary_threshold))

    # load embeddings and target words
    src_emb, trg_emb = load_word_vectors(src_emb_path, trg_emb_path)
    target_words_dict, target_words = load_target_words(target_words_path, load_labels=False)

    run_transform = False
    if xform is None:
        # transformation matrix
        xform = os.path.basename(src_emb_path) + '-' + os.path.basename(trg_emb_path) + '.xform'
        xform = os.path.join(TMP_DIR, xform)
        run_transform = True


    # file with results
    output_file = os.path.basename(src_emb_path) + '-' + os.path.basename(trg_emb_path) + '.ranks'
    output_file = os.path.join(TMP_DIR, output_file)

    output_file_binary = os.path.basename(src_emb_path) + '-' + os.path.basename(trg_emb_path) + '.binary'
    output_file_binary = os.path.join(TMP_DIR, output_file_binary)


    if run_transform is True :
        trans_matrix,trans_matrix2,testw = build_transform(src_emb,trg_emb,target_words_dict, 
        src_emb_path=src_emb_path,
        trg_emb_path=trg_emb_path,
        max_links = max_links,
        xform=xform)

    # similarities used for generating output file
    rank_similarities = []

    # original similarities
    similarities_unchanged = []
    similarities_to_orig_word = []
    similarities_to_trans_vec = []
    binar_change = []
    neighbrs_inter_sizes = []


    for target_word in target_words:
        # print("Word:" + str(target_word), end='')
        sim, sim_to_orig_word, sim_to_trans_vec = compare_sense(target_word, src_emb, trg_emb, trans_matrix, topn=topn, individual_xform=individual_xform, transformation_matrix2=trans_matrix2)

        # compute intersection of nearest neigbhrs
        neighbrs_inter_sizes.append(compute_inter_size(sim_to_orig_word, sim_to_trans_vec))

        similarities_unchanged.append(sim)
        similarities_to_orig_word.append(sim_to_orig_word)
        similarities_to_trans_vec.append(sim_to_trans_vec)

        if one_minus is True:
            sim = 1 - sim
        rank_similarities.append(sim)

    binary_threshold = None
    min_neighbours_count = None

    # check for presence of wordsout file, and build it if necessary
    if use_nearest_neigbhrs is True and use_binary_threshold is True:
        raise Exception("I can compute only one at once")

    #     druha nejvetsi hodnota, pokud licha tak +1 a tu vydelim dvema
    # default
    # en - 62, vzal sem 31
    # de - 38, vzal jsem 19
    # la - 39, vzal jsem 18 -- prolbem, ovlivni vysledek  asi]
    # swe - 35, vzal jsem 17
    # u latiny to bylo sice 39 ale ja vzal 18 tj. muselo by to byt 36
    # u svedstiny to bylo 35 ale vzal jsem 17
    # default reversed
    # en - 62, vzal jsem 31
    # de - 39, vzal jsem 19
    # la - 61, vzal jsem 30
    # sw - 41, vzal jsem 20

    if use_nearest_neigbhrs is True:
        print("Computing decide_binary_neighbours")
        # if the max number is there two times we still take the second largest value
        set_list = set(neighbrs_inter_sizes)
        set_list.remove(max(set_list))
        second_largest = int(max(set_list)/2)
        min_neighbours_count = second_largest
        print("Second largest is:" + str(second_largest))


    # compute average similarity which will be the threshold
    if use_binary_threshold is True:
        print("Computing binary_threshold")
        avg_sim = np.average(similarities_unchanged)
        avg_sim = round(avg_sim, 3)
        print('similarity average:' + str(avg_sim))
        binary_threshold = avg_sim


    # todo mozna zkusit prumernou podobnost k-nejblizsisch slov, zkusit na durelu

    # iterate again over words and compute binary task
    for target_word, sim, sim_to_orig_word, sim_to_trans_vec, nearest_neigbh_size in zip(
            target_words, similarities_unchanged, similarities_to_orig_word, similarities_to_trans_vec, neighbrs_inter_sizes):
        if use_binary_threshold is True:
            binar_change.append(decide_binary_change_threshold(sim, binary_threshold))

        if use_nearest_neigbhrs is True:
            binar_change.append(decide_binary_neighbours(nearest_neigbh_size, min_neighbours_count))


    # write to tmp folder
    with open(output_file, 'w') as f:
        for word, sim in zip(target_words, rank_similarities):
            f.write(word + '\t' + str(sim) + '\n')

    # write binary predictions to tmp folder
    with open(output_file_binary, 'w') as f:
        for word, clazz in zip(target_words, binar_change):
            f.write(word + '\t' + str(clazz) + '\n')

    if save_file_ranks is not None:
        with open(save_file_ranks, 'w') as f:
            for word, sim in zip(target_words, rank_similarities):
                f.write(word + '\t' + str(sim) + '\n')

    # save binary predictions
    if save_file_binary is not None:
        with open(save_file_binary, 'w') as f:
            for word, clazz in zip(target_words, binar_change):
                f.write(word + '\t' + str(clazz) + '\n')

    # write summary of embedding and transform statistics to the general folder
    save_summary( src_emb, trg_emb, trans_matrix, testw, topn, target_words, rank_similarities)

    if run_spearman is True:
        rho, pval = compute_spearman(gold_file_task2, output_file, print_res=False)
        acc = accuracy_official(gold_file_task1, output_file_binary)
        # print("task1 \t task2")
        # print(str(acc), str(rho))

        # save these statistics:
        with open('data/tmp_test_results/'+general_folder+'/xform_stats.txt','a') as fi:
            print('Outputs=',round(rho,3), round(acc,3), binary_threshold, min_neighbours_count, file=fi)
        return round(rho,3), round(acc,3), binary_threshold, min_neighbours_count
    return 0, 0, binary_threshold, min_neighbours_count


def save_summary(src_emb, trg_emb, trans_matrix, testw, topn, target_words, rank_similarities):
    """
    src_emb is the "source embedding" a semantic vector space
    trg_emb is the "target embedding", another one
    trans_matrix is a linear transform from src_emb to trg_emb
    and testw is a list of upto 100 words.
    topn is a number passed as an argument to compare(), default 100
    target_words is a list of the target words

    create a file, xform_stats, next to answers folder in general directory
    save there some statistics about the xform and the embeddings, including:
        the norm of the xform, which should be near zero
        an average estimate of near neighbor overlaps

    The file is also written in other places...
    """
    global general_folder

    target_d = dict(zip(target_words,rank_similarities))
    outpath_s = 'data/tmp_test_results/'+general_folder + '/xform_stats.txt'
    with open(outpath_s,'a') as fo:
        sumscore = [0]*3
        for w in it.chain(testw,target_words):
          sets = [set() for i in range(3)]
          for i,(emb,vec) in enumerate([(src_emb,src_emb[w]),(trg_emb,trg_emb[w]),(trg_emb,src_emb[w] @ trans_matrix)]):

            # find nearest neighbors list for w in emb
            slist = emb.wv.similar_by_vector(vec,  topn=topn)
            # remove hubs and other neighbors for whom w is not in their nearest list
            for v,d in slist:
                vlist = {x:d for (x,d) in emb.wv.most_similar(positive=[v], negative=[], topn=topn)}
                if w in vlist:  sets[i].add(v)

            # if this is a transformed target word (i==1) check rank of target
            # that is, how many otherwords are closer to here than trg_emb[w]
            if i == 2 : #and w in target_d:
                similarity = np.dot(gma.unitvec(vec),
                                    gma.unitvec(trg_emb[w]))
                neighbor = trg_emb.similar_by_vector(vec,
                                        topn=2000,restrict_vocab=100000)
                if similarity < neighbor[1999][1]:
                    rank = 2000
                else:
                    hin = 1999
                    lin = 0
                    rank = None
                    while hin > lin:
                        if abs(neighbor[hin][1] - similarity) < 1e-5:
                            rank = hin
                            break
                        if abs(neighbor[lin][1] - similarity) < 1e-5:
                            rank = lin
                            break
                        mid = (hin+lin)//2
                        if abs(neighbor[mid][1] - similarity) < 1e-5:
                            rank = mid
                            break
                        if neighbor[mid][1] < similarity:
                            hin = mid
                        elif lin == mid: 
                            rank = lin
                            break
                        else:
                            lin = mid

                #print('r('+w+')', rank)
                #print('r('+w+')', rank, file=fo)

            # compute size of intersection of src and trf nn lists, |intersect|/|union|
          wc = ''
          for j,(left,right) in enumerate([(0,1),(1,2),(0,2)]):
            sizeS = len(sets[left])
            sizeT = len(sets[right])
            sizeI = len(sets[left].intersection(sets[right]))
            denom = sizeS+sizeT-sizeI
            if denom == 0:
                fracI = 0
            else:
                fracI = sizeI/denom
            # sum them up
            if w in target_d:
                wc += ' '+str(fracI) 
            else:
                wc += ' '+str(fracI) # print stats on each testw, also
                sumscore[j] += fracI # but don't accumulate for the target words

          if w in target_d:
              print('r('+w+')*', rank)
              print('r('+w+')*', rank, file=fo)
              print('t('+w+')*', wc)
              print('t('+w+')*', wc, file = fo)
              print('d('+w+')*', target_d[w])
              print('d('+w+')*', target_d[w], file = fo)
          else:
              print('r('+w+')', rank)
              print('r('+w+')', rank, file=fo)
              print('t('+w+')', wc)
              print('t('+w+')', wc, file = fo)
              print('d('+w+')', 1-similarity)
              print('d('+w+')', similarity, file=fo)

            
        ss = [str(s/len(testw)) for s in sumscore]
        print('Neighborliness:', ' '.join(ss))
        print('Neighborliness:', ' '.join(ss), file=fo)


        



def trans_arrays(src_emb, trg_emb, transform_set):
    """
    Turn transform_set into arrays for generating or checking transform
    """
    if transdict_length:
        halfway = transdict_length//2
    else:
        halfway = len(transform_set)//2
    if transdict_length == None or transdict_length > len(transform_set):
        X = np.ndarray((len(transform_set),src_emb.vectors.shape[1]),
                            dtype=np.float32)
        Y = np.ndarray((len(transform_set),trg_emb.vectors.shape[1]),
                            dtype=np.float32)
        for i,x in zip(range(1000000),transform_set):
            if i==halfway: print('transdict['+str(i)+']',x)
            X[i] = src_emb[x]
            Y[i] = trg_emb[x]
        return X,Y
    else: # build a limited transdict.  Implementation favors src_emb
        # assumes that order of words in index2entity follows corpus frequency
        # discards very early words (maybe too many?) in favor of early ones
        total = 0
        two = 3 # max allowed ratio between src and trg ranks
        base = min(transdict_base,len(transform_set)-transdict_length)
        X = np.zeros((transdict_length,src_emb.vectors.shape[1]),
                            dtype=np.float32)
        Y = np.zeros((transdict_length,src_emb.vectors.shape[1]),
                            dtype=np.float32)
        trg_rank = {w:r for r,w in enumerate(trg_emb.index2entity)}
        if (transdict_length != None and 
            transdict_length < len(transform_set)):
            #prob of inclusion maybe too high.  I assume words less<base occur
            # in both corpora.  Could compute correct value with a loop:
            # len([w for w in ...
            pink =  (transdict_length)/len(transform_set) * 0.5 +0.5
        else: pink = 1.0
        rand = random.Random()
        with open('data/tmp_test_results/'+general_folder+'/xform_stats.txt','w') as fo:
            testw = []
            for i,x in enumerate(it.islice(src_emb.index2entity,base,None)):
                if pink != 1.0 and rand.uniform(0,1.0) > pink: continue
                if x in transform_set:
                    xr = i+base
                    yr = trg_rank[x]
                    if yr < xr//two or yr > xr*two: continue
                    lxr = xr
                    if len(testw) < 100 and rand.uniform(0,1.0) < 0.04:
                        testw.append(x)
                        continue
                    X[total] = src_emb[x]
                    Y[total] = trg_emb[x]
                    total += 1
                    if total==halfway: print('transdict['+str(total)+']',x)
                    if total >= transdict_length: break
            if total < transdict_length:
                print('transdict_total:' , total,'max src rank:',lxr)
                print('transdict_total:' , total,'max src rank:',lxr, file=fo)
            print('\ntransdict_total:' , total,'max src rank:',lxr, file=fo)
            print('TestWords:',testw, file=fo)
        return X[:total,:],Y[:total,:],testw


def build_transform(src_emb, trg_emb, target_words_dict, target_word= None, 
        src_emb_path=None,
        trg_emb_path=None,
        max_links = None,
        xform=None):
    """
    build a transform from the src embedding space to the trg embedding space,
    using the parameters 
        src_emb and trg_emb are gensim.KeyedVector
        target_word_dict is a dict of target words
        target_word, if not None: build custom xform for target word
    and a several global parameters:
       ver_xform   -- either 'java' : use Tomas CCA java code
                          or 'python': import and use python port
       individual_xform-- True or False.  different xform for each word
       transdict_length -- None or integer -- limit size of translation dict 
    """
    if target_word == None:
            transform_set = build_transform_set(src_emb,trg_emb,
                                     target_words_dict, transdict_limit)
    else:
            transform_set = build_transform_set_individual(src_emb, trg_emb,
                                     target_word, transdict_length)


    if ver_xform == 'java':

            trans_dict_path = os.path.join(TMP_DIR, 'trans.dict')
            halfway = len(transform_set)//2
            with open(trans_dict_path,'w') as fo:
                for i,x in enumerate(transform_set):
                    if i==halfway: print('transdict['+str(i)+']',x)
                    fo.write(x+'\t'+x+'\n')

            if sys.platform == 'linux':
                cpsep = ':'
            else:
                cpsep = ";"
            # but even with cpsep, this command is system-dependent...

            exit_code = os.spawnvp(os.P_WAIT,
              # '/usr/lib/jvm/java-11-openjdk-amd64/bin/java', 
              '/usr/jdk-9.0.4/bin/java',
               ['-Xms6000000000', '-cp',
               "CrossLingualSemanticSpaces-0.0.1-SNAPSHOT-jar-with-dependencies.jar" +
                                       cpsep + "CCAjar.jar", 'clss.CCA',
                                       src_emb_path, trg_emb_path,
                                       trans_dict_path, xform,  str(max_links)])
            if exit_code != 0:
                print('?exit_code from java=', exit_code, file=sys.stderr)
                sys.exit(exit_code)

            trans_matrix = load_transform_matrix(xform)
            if compute_trans_qual:
                X,Y,testw = trans_arrays(src_emb,trg_emb,transform_set)
                with open('data/tmp_test_results/'+general_folder+'/xform_stats.txt','a') as fi:
                    if False: #thirdspace:
                        print('2-norm xform=', trans_diff2(X,Y,trans_matrix,trans_matrix2))
                        print('2-norm xform=', trans_diff2(X,Y,trans_matrix,trans_matrix2),file=fi)
                    else:
                        print('2-norm xform=', trans_diff(X,Y,trans_matrix))
                        print('2-norm xform=', trans_diff(X,Y,trans_matrix),file=fi)

            return trans_matrix,None , testw #java code not ready for thirdspace

    else: # set up trans_matrix with python code:
            import ccaxform1 as cc
            #x,y = cc.build_dict(target_words_dict, src_emb, trg_emb)
            """lx, ly = [],[]
            for x in src_emb.vocab.keys():
                if not x in trg_emb.vocab or x in target_words_dict: continue
                lx.append(src_emb[x])
                ly.append(trg_emb[x])
            X = np.ndarray((len(lx),len(lx[0])),dtype = np.float32)
            Y = np.ndarray((len(ly),len(ly[0])),dtype = np.float32)
            """
            X,Y,testw = trans_arrays(src_emb,trg_emb,transform_set)
            if thirdspace:
                trans_matrix,trans_matrix2 = cc.transforms(X,Y)
            else:
                trans_matrix2 = None
                trans_matrix = cc.transform(X,Y)
            if compute_trans_qual:  
                with open('data/tmp_test_results/'+general_folder+'/xform_stats.txt','a') as fo:
                    if thirdspace:
                        print('2-norm xform=', trans_diff2(X,Y,trans_matrix,trans_matrix2))
                        print('2-norm xform=', trans_diff2(X,Y,trans_matrix,trans_matrix2),file=fo)
                    else:
                        print('2-norm xform=', trans_diff(X,Y,trans_matrix))
                        print('2-norm xform=', trans_diff(X,Y,trans_matrix),file=fo)
            return trans_matrix, trans_matrix2, testw

def trans_diff(X,Y,trans_matrix):
    """
    Only a tiny piece of code, but I do use it twice.
    """
    T = (X @ trans_matrix)
    D = T - Y
    D2 = D * D  # this is a broadcast, i.e. element by element, multiply
    q = np.sum(D2)
    return q/X.shape[0]/X.shape[1]

def trans_diff2(X,Y,trans_matrix,trans_matrix2):
    """
    Here we get the sum-squared-differences in the thirdspace, instead of target
    This is used when we use canonical-correlation transforms to intermediate
    space, instead of using T1 @ pinv(T2) to transform source to target.
    """
    T = (X @ trans_matrix)
    S = (Y @ trans_matrix2)
    D = T - S
    D2 = D * D  # this is a broadcast, i.e. element by element, multiply
    q = np.sum(D2)
    return q/X.shape[0]/trans_matrix.shape[1]



def compute_spearman_between_res():
    t = TEST_DATA_RESULTS_DIR
    from os.path import join
    # tasks_paths = [join(t, 'default'), join(t, 'default_reversed'),
    #                join(t, 'default_binary_threshold'), join(t, 'default_reveresed_binary_threshold')]
    tasks_paths = [join(t, 'default'), join(t, 'default_reversed'), join(t, 'LDA-100'), join(t, 'map-ort-i'), join(t, 'map-unsup')]
    tasks_paths = [join(path, 'answer', 'task2') for path in tasks_paths]

    tasks_paths_english = [join(path, 'english.txt') for path in tasks_paths]
    tasks_paths_german = [join(path, 'german.txt') for path in tasks_paths]
    tasks_paths_latin = [join(path, 'latin.txt') for path in tasks_paths]
    tasks_paths_swedish = [join(path, 'swedish.txt') for path in tasks_paths]

    tasks_tuples = [('English', tasks_paths_english), ('German', tasks_paths_german), ('Latin', tasks_paths_latin), ('Swedish', tasks_paths_swedish)]

    for (lang, paths_list) in tasks_tuples:
        print('Computing correlation between our results for ' + lang)
        for base_path in paths_list:
            print('Solution:'+ str(base_path.split('/')[-4]))
            print('#####')
            for tmp_path in paths_list:
                print(str(tmp_path.split('/')[-4]))
                compute_spearman(base_path, tmp_path)
                print('----------------')

            print('-------------------------------')
        print('#################################')
        print('#################################')
        print('#################################')
    pass


def compute_inter_size(sim_to_orig_word, sim_to_trans_vec):
    orig_words = [tup[0] for tup in sim_to_orig_word]
    trans_words = [tup[0] for tup in sim_to_trans_vec]

    orig_words = set(orig_words)
    trans_words = set(trans_words)

    inters = orig_words.intersection(trans_words)
    inter_size = len(inters)
    # print(" inter size:" + str(inter_size))

    return inter_size


def decide_binary_neighbours(nearest_neigbh_size, min_neighbours_count):

    if nearest_neigbh_size >= min_neighbours_count:
        return 0
    else:
        return 1



def decide_binary_change_threshold(similarity, threshold):
    # print(" sim:" + str(similarity))
    if similarity >= threshold:
        return 0
    else:
        return 1


def compute_spearman(file_gold_path, file_pred_path, print_res=True):
    gold_words_dict, _ = load_target_words(file_gold_path)
    pred_words_dict, _ = load_target_words(file_pred_path)

    if(len(gold_words_dict) != len(pred_words_dict)):
        raise Exception("Word dictionaries do not match")

    gold_list = list(gold_words_dict.keys())
    gold_list.sort()
    pred_list = list(pred_words_dict.keys())
    pred_list.sort()

    if len(gold_list) != len(pred_list):
        print(len(pred_list), '!=', len(pred_list))
        raise Exception("Word dictionaries do not match")

    ranks_gold = []
    ranks_pred = []

    for gold, pred in zip(gold_list, pred_list):
        ranks_gold.append(float(gold_words_dict[gold]))
        ranks_pred.append(float(pred_words_dict[pred]))

    rho, pval = ss.spearmanr(ranks_gold, ranks_pred)
    if print_res:
        print('Rho:' + str(rho) + ' p-value:' + str(pval))

    return rho, pval


def delete_tmp_dir():
    tmp_dir = TMP_DIR
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(TMP_DIR)

def build_transform_dict(src_emb, trg_emb, trans_dict_path, target_words_dict):
    transform_set = build_transform_set(src_emb_trg_emb, target_words_dict)

    with open(trans_dict_path, 'w', encoding='utf-8') as f:
        for word in transform_set:
            if not word.strip():
                continue

            f.write(word + '\t' + word + '\n')

def build_transform_set(src_emb, trg_emb, target_words_dict, size_limit = None):
    """
    build a set of guide words for the trans.dict,
    which contains only those words which appear in both embeddings and
    are not target words.
    """
    src_vocab = set(it.islice(src_emb.index2entity,transdict_base,size_limit))
    trg_vocab = set(it.islice(trg_emb.index2entity,transdict_base,size_limit))
    target_vocab = set(target_words_dict)

    src_vocab.intersection_update(trg_vocab)
    src_vocab.difference_update( target_vocab ) # remove elements
    #if size_limit != None:
    #    while len(src_vocab) > size_limit:
    #        src_vocab.pop() # remove one element
    return src_vocab

def build_transform_set_individual(src_emb, trg_emb, target_word, size_limit = None):
    """
    build a transform dictionary set of words, the 'neighborhood'
    which have the same approx distance from the target word in both embeddings
    """
    ssize = src_emb.vectors.shape[0]//2  # neighborhood should not include 
    tsize = trg_emb.vectors.shape[0]//2  # more than half the vocabulary
    if size_limit == None:
        size_limit = min(ssize,tsize)
    else:
        size_limit = min(ssize,tsize,size_limit)

    retval = set()
    slist = src_emb.most_similar(positive = [target_word], topn = 2*size_limit)
    tlist = trg_emb.most_similar(positive = [target_word], topn = 2*size_limit)

    retval = set([x[0] for x in slist]).intersection(set([x[0] for x in tlist]))
    sfurthest = tfurthest = None
    for i in range(2*size_limit-1):
        if slist[-i][0] in retval:
            sfurthest = slist[-i][1]
            break
    for i in range(2*size_limit-1):
        if tlist[-i][0] in retval:
            tfurthest = tlist[-i][1]
            break
    print (target_word, 'transform set size', len(retval), sfurthest,tfurthest)

    """
        In the following loop, slist and tlist are sorted by similarity; 
        The nearest item is the target_word, at slist[0] and tlist[0]
        we want to consider matching only items in the same "band" of
        concentric circles around the target word.  The intersections of the
        bands in tlist and slist become part of the local translation dictionary
        for the target word.  We don't want to include nearby words in different
        bands, because if they have different relations to the target word in
        the two spaces then transforming them to the same location might
        somehow diminish differences in senses.

        The concentric squares/circles idea is too two-dimensional; in n-space, 
        the volume of each band should be most**n - least**n.  The 2-d bands
        may be very narrow!
    
    increment,least,most = 100,0,0
    ss = set()  #set of items in slist[least:most+1]
    st = set()
    for i,(sl,tl ) in enumerate(zip(slist,tlist)):
        if  i <  most:
            ss.add(sl[0])
            st.add(tl[0])
        else:  #  i == most, finish up band
            ss.add(sl[0])
            st.add(tl[0])
            #debugging printout:
            print(' measure',target_word,most, sl,tl)
            ss.intersection_update(st)
            retval.update(ss)
            if sl[1] < 0.01 or tl[1] < 0.01: #too distant for "neighborhood"
                        break
            #else: # don't add target word to retval; 
            increment,least,most = increment+200, most+1, most+increment

            ss = set()
            st = set()
                            
    print('size of transdict',len(retval),'for',target_word) # debugging
    """
    return retval
 
def load_target_words(target_words_path, load_labels=True):
    with open(target_words_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    trg_dict = {}
    trg_words = []

    for line in lines:
        word = line.split()[0]
        if load_labels is True:
            label = line.split()[1]
        else:
            label = 0.5
        word = word.strip()
        trg_words.append(word)
        trg_dict[word] = label

    # print('Loaded :' + str(len(trg_words)) + ' target words from file:' + str(target_words_path))

    return trg_dict, trg_words


# for do_compare, I wrote the models in binary format.  So load them thus
def load_word_vectors(src_file_path, trg_file_path):
    #src_emb = KeyedVectors.load_word2vec_format(src_file_path, binary=False)
    src_emb = KeyedVectors.load(src_file_path)#, binary=False)
    normalize(src_emb)
    trg_emb = KeyedVectors.load(trg_file_path)#, binary=False)
    normalize(trg_emb)

    return src_emb, trg_emb

def normalize(emb):
    """
    emb is a gensim.Word2VectKeyedVectors object.
    it has vectors and vectors_norm members, and we will initialize both to 
    the same value, as in gensim.models.Word2VectKeyedVectors._init_sim()
    """
    if normalization == 0: return #global variable...
    elif normalization & 1 != 0:  # center 
        origin = emb.vectors.sum(0) / emb.vectors.shape[0]
        emb.vectors_norm = emb.vectors = emb.vectors - origin
    elif normalization & 2 != 0:  # unit normalize
        # following code from gensim.models.keyedvectors.py init_sim
        dist = np.sqrt((emb.vectors ** 2).sum(-1))[..., np.newaxis]
        emb.vectors /= dist

        # following is previous version, 14X slower.
        #for i,v in enumerate(emb.vectors):
        #    x = math.sqrt(v.dot(v))
        #    emb.vectors[i] = v/x

        emb.vectors_norm = emb.vectors

def init_folders(general_dir):
    task_1 = os.path.join(TEST_DATA_RESULTS_DIR, general_dir, 'answer', 'task1')
    task_2 = os.path.join(TEST_DATA_RESULTS_DIR, general_dir, 'answer', 'task2')

    zip_file = 'UWB_' + general_dir
    zip_file = os.path.join(TEST_DATA_RESULTS_DIR, zip_file)

    folder_to_zip = os.path.join(TEST_DATA_RESULTS_DIR, general_dir)
    pathlib.Path(task_1).mkdir(parents=True, exist_ok=True)
    pathlib.Path(task_2).mkdir(parents=True, exist_ok=True)

    return task_1, task_2, folder_to_zip, zip_file


def zip_folder(folder_to_zip, zip_file):
    shutil.make_archive(zip_file, 'zip', folder_to_zip)

if __name__ == '__main__':
    main()
