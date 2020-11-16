import os


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_PATH, 'data')
EXPORT_DIR = os.path.join(DATA_DIR, 'export')

TMP_DIR = os.path.join(DATA_DIR, 'tmp')


ROYAL_SOCIETY_FOLDER = os.path.join(DATA_DIR, 'Royal_Society')
ROYAL_SOCIETY_DATA_DESC = os.path.join(ROYAL_SOCIETY_FOLDER, 'Royal_Society_Corpus_v4.0.1_meta.tsv')
ROYAL_SOCIETY_DATA_FOLDER = os.path.join(ROYAL_SOCIETY_FOLDER, 'Royal_Society_Corpus_v4.0.1_texts_txt')
ROYAL_SOCIETY_EXPORT_PATH = os.path.join(EXPORT_DIR, 'royal_society.tsv')
ROYAL_SOCIETY_EXPORT_PREPROCESSED_PATH = os.path.join(EXPORT_DIR, 'royal_society-prep.tsv')

CLMET_DATA_FOLDER = os.path.join(DATA_DIR, 'CLMET3')
CLMET_EXPORT_PATH = os.path.join(EXPORT_DIR, 'clmet3.tsv')
CLMET_EXPORT_PREPROCESSED_PATH = os.path.join(EXPORT_DIR, 'clmet3-prep.tsv')

UDPIPE_MODELS_DIR = os.path.join(DATA_DIR,'udpipe-models')
UDPIPE_MODEL_PATH = os.path.join(UDPIPE_MODELS_DIR,'english-ewt-ud-2.4-190531.udpipe')

CLMET_SPLITTED_T1_PATH = os.path.join(EXPORT_DIR, 'clmet3-')
CLMET_SPLITTED_ROYAL_PATH = os.path.join(EXPORT_DIR, 'royal-')

EMBEDDINGS_EXPORT_PATH = os.path.join(DATA_DIR, 'embedding_export')

# EMBEDDINGS_EXPORT_PATH = os.path.join(EMBEDDINGS_EXPORT_PATH, 'fasttext')

SENSE_WORDS_PATH = os.path.join(DATA_DIR, 'sense_words')

TRANSFORMATION_PATH = os.path.join(DATA_DIR, 'transformation')

TEST_DATA_PATH = os.path.join(DATA_DIR, 'test-data')
TEST_DATA_RESULTS_DIR = os.path.join(DATA_DIR, 'tmp_test_results')

ENGLISH_TEST_CORPUS_1 = os.path.join(TEST_DATA_PATH, 'english', 'semeval2020_ulscd_eng', 'corpus1', 'lemma', 'ccoha1.txt')
ENGLISH_TEST_CORPUS_2 = os.path.join(TEST_DATA_PATH, 'english', 'semeval2020_ulscd_eng', 'corpus2', 'lemma', 'ccoha2.txt')
ENGLISH_TEST_TARGET_WORDS = os.path.join(TEST_DATA_PATH, 'english','semeval2020_ulscd_eng','targets.txt')


GERMAN_TEST_CORPUS_1 = os.path.join(TEST_DATA_PATH, 'german', 'semeval2020_ulscd_ger', 'corpus1', 'lemma', 'dta.txt')
GERMAN_TEST_CORPUS_2 = os.path.join(TEST_DATA_PATH, 'german', 'semeval2020_ulscd_ger', 'corpus2', 'lemma', 'bznd.txt')
GERMAN_TEST_TARGET_WORDS = os.path.join(TEST_DATA_PATH, 'german', 'semeval2020_ulscd_ger', 'targets.txt')

LATIN_TEST_CORPUS_1 = os.path.join(TEST_DATA_PATH, 'latin', 'semeval2020_ulscd_lat', 'corpus1', 'lemma', 'LatinISE1.txt')
LATIN_TEST_CORPUS_2 = os.path.join(TEST_DATA_PATH, 'latin', 'semeval2020_ulscd_lat', 'corpus2', 'lemma', 'LatinISE2.txt')
LATIN_TEST_TARGET_WORDS = os.path.join(TEST_DATA_PATH, 'latin','semeval2020_ulscd_lat', 'targets.txt')


SWEDISH_TEST_CORPUS_1 = os.path.join(TEST_DATA_PATH, 'swedish', 'semeval2020_ulscd_swe', 'corpus1', 'lemma', 'kubhist2a.txt')
SWEDISH_TEST_CORPUS_2 = os.path.join(TEST_DATA_PATH, 'swedish', 'semeval2020_ulscd_swe', 'corpus2', 'lemma', 'kubhist2b.txt')
SWEDISH_TEST_TARGET_WORDS = os.path.join(TEST_DATA_PATH, 'swedish', 'semeval2020_ulscd_swe', 'targets.txt')


POST_EVAL_DATA = os.path.join(DATA_DIR, 'post_eval_data')
TEST_DATA_TRUTH_ANSWER_TASK_1 = os.path.join(POST_EVAL_DATA, 'test_data_truth', 'task1')
TEST_DATA_TRUTH_ANSWER_TASK_2 = os.path.join(POST_EVAL_DATA, 'test_data_truth', 'task2')


ENGLISH_TEST_GOLD_TASK_1 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_1, 'english.txt')
ENGLISH_TEST_GOLD_TASK_2 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_2, 'english.txt')

GERMAN_TEST_GOLD_TASK_1 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_1, 'german.txt')
GERMAN_TEST_GOLD_TASK_2 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_2, 'german.txt')


LATIN_TEST_GOLD_TASK_1 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_1, 'latin.txt')
LATIN_TEST_GOLD_TASK_2 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_2, 'latin.txt')

SWEDISH_TEST_GOLD_TASK_1 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_1, 'swedish.txt')
SWEDISH_TEST_GOLD_TASK_2 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_2, 'swedish.txt')


"""
In following lines I override some of the paths defined above
"""

EVAL_SEM = '/home/staylor/spring20/evalSem'
JUNQUE = os.path.join(EVAL_SEM, 'junque.txt') # a quite short file...

ENGLISH_TEST_CORPUS_1 = os.path.join(EVAL_SEM, 'English-eval', 'cr1')
ENGLISH_TEST_CORPUS_2 = os.path.join(EVAL_SEM, 'English-eval', 'cr2')
ENGLISH_TEST_TARGET_WORDS = os.path.join(EVAL_SEM, 'English-eval', 'targets.txt')


GERMAN_TEST_CORPUS_1 = os.path.join(EVAL_SEM, 'German-eval', 'cr1')
GERMAN_TEST_CORPUS_2 = os.path.join(EVAL_SEM, 'German-eval', 'cr2')
GERMAN_TEST_TARGET_WORDS = os.path.join(EVAL_SEM, 'German-eval', 'targets.txt')

LATIN_TEST_CORPUS_1 = os.path.join(EVAL_SEM, 'Latin-eval', 'cr1')
LATIN_TEST_CORPUS_2 = os.path.join(EVAL_SEM, 'Latin-eval', 'cr2')
LATIN_TEST_TARGET_WORDS = os.path.join(EVAL_SEM, 'Latin-eval', 'targets.txt')

SWEDISH_TEST_CORPUS_1 = os.path.join(EVAL_SEM, 'Swedish-eval', 'cr1')
SWEDISH_TEST_CORPUS_2 = os.path.join(EVAL_SEM, 'Swedish-eval', 'cr2')
SWEDISH_TEST_TARGET_WORDS = os.path.join(EVAL_SEM, 'Swedish-eval', 'targets.txt')


ENGLISH_TEST_GOLD_TASK_1 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_1, 'english.txt')
ENGLISH_TEST_GOLD_TASK_2 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_2, 'english.txt')

GERMAN_TEST_GOLD_TASK_1 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_1, 'german.txt')
GERMAN_TEST_GOLD_TASK_2 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_2, 'german.txt')

LATIN_TEST_GOLD_TASK_1 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_1, 'latin.txt')
LATIN_TEST_GOLD_TASK_2 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_2, 'latin.txt')

SWEDISH_TEST_GOLD_TASK_1 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_1, 'swedish.txt')
SWEDISH_TEST_GOLD_TASK_2 = os.path.join(TEST_DATA_TRUTH_ANSWER_TASK_2, 'swedish.txt')
