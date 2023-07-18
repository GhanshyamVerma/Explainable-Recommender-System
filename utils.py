from __future__ import absolute_import, division, print_function

import sys
import random
import pickle
import logging
import logging.handlers
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
import torch


# Dataset names.

CHALLENGE = 'challenge'

# Dataset directories.
DATASET_DIR = {
    CHALLENGE: './Dataset/Challenge_Dataset'
}

# Model result directories.
TMP_DIR = {
    CHALLENGE: './tmp/Challenge_Dataset'
}

# Label files.
LABELS = {
    CHALLENGE: (TMP_DIR[CHALLENGE] + '/train_label.pkl', TMP_DIR[CHALLENGE]+'/test_label.pkl')
}


# Entities
USER = 'user'
ARTICLE = 'article'
WORD = 'word'
RARTICLE = 'related_article'
TOPIC = 'topic'
PRODUCT = 'product'
TOPIC_TAG = 'topic_tag'
PRODUCT_TAG = 'product_tag'



# Relations
#RESPONSE = 'response'
RECOMMENDED = 'recommended'
WITHIN = 'within' #similar to the user mention words, word within article
HAS_TOPIC = 'has_topic'
HAS_PRODUCT = 'has_product'
HAS_TOPIC_TAG = 'has_topic_tag'
HAS_PRODUCT_TAG = 'has_product_tag'
ALSO_RESPONSE = 'also_response'
RESPONSE_TOGETHER = 'response_together'
RECOMMENDED_TOGETHER = 'recommended_together'
#HAS_FULL_TEXT = 'has_full_text'
SELF_LOOP = 'self_loop'  # only for kg env

KG_RELATION = {
    USER: {
        #RESPONSE: ARTICLE,
        RECOMMENDED: ARTICLE,
    },
    WORD: {
        WITHIN: ARTICLE,
    },
    ARTICLE: {
        #RESPONSE: USER,
        RECOMMENDED: USER,
        HAS_TOPIC: TOPIC,
        HAS_PRODUCT: PRODUCT,
        HAS_TOPIC_TAG: TOPIC_TAG,
        HAS_PRODUCT_TAG: PRODUCT_TAG,
        ALSO_RESPONSE: RARTICLE,
        RESPONSE_TOGETHER: RARTICLE,
        RECOMMENDED_TOGETHER: RARTICLE,
        WITHIN: WORD,
    },
    TOPIC: {
        HAS_TOPIC: ARTICLE,
    },
    PRODUCT: {
        HAS_PRODUCT: ARTICLE,
    },
    RARTICLE: {
        ALSO_RESPONSE: ARTICLE,
        RESPONSE_TOGETHER: ARTICLE,
        RECOMMENDED_TOGETHER: ARTICLE,
    },
    TOPIC_TAG: {
        HAS_TOPIC_TAG: ARTICLE,
    },
    PRODUCT_TAG: {
        HAS_PRODUCT_TAG: ARTICLE,
    }
}


PATH_PATTERN = {
    11: ((None, USER), (RECOMMENDED, ARTICLE), (RECOMMENDED, USER),(RECOMMENDED,ARTICLE)),
    13: ((None, USER), (RECOMMENDED, ARTICLE), (HAS_TOPIC, TOPIC), (HAS_TOPIC, ARTICLE)),
    14: ((None, USER), (RECOMMENDED, ARTICLE), (HAS_PRODUCT, PRODUCT), (HAS_PRODUCT, ARTICLE)),
    15: ((None, USER), (RECOMMENDED, ARTICLE), (ALSO_RESPONSE, ARTICLE), (ALSO_RESPONSE, ARTICLE)),
    16: ((None, USER), (RECOMMENDED, ARTICLE), (RESPONSE_TOGETHER, ARTICLE), (RESPONSE_TOGETHER, ARTICLE)),
    17: ((None, USER), (RECOMMENDED, ARTICLE), (RECOMMENDED_TOGETHER, ARTICLE), (RECOMMENDED_TOGETHER, ARTICLE)),
    18: ((None, USER), (RECOMMENDED, ARTICLE),(RECOMMENDED, USER),(RECOMMENDED, ARTICLE)),
    19: ((None, USER), (RECOMMENDED, ARTICLE), (HAS_TOPIC_TAG, TOPIC_TAG),(HAS_TOPIC_TAG, ARTICLE)),
    20: ((None, USER), (RECOMMENDED, ARTICLE), (HAS_PRODUCT_TAG, PRODUCT_TAG),(HAS_PRODUCT_TAG, ARTICLE)),
}


def get_entities():
    return list(KG_RELATION.keys())


def get_relations(entity_head):
    return list(KG_RELATION[entity_head].keys())


def get_entity_tail(entity_head, relation):
    return KG_RELATION[entity_head][relation]


def compute_tfidf_fast(vocab, docs):
    """Compute TFIDF scores for all vocabs.

    Args:
        docs: list of list of integers, e.g. [[0,0,1], [1,2,0,1]]

    Returns:
        sp.csr_matrix, [num_docs, num_vocab]
    """
    # (1) Compute term frequency in each doc.
    data, indices, indptr = [], [], [0]
    for d in docs:
        term_count = {}
        for term_idx in d:
            if term_idx not in term_count:
                term_count[term_idx] = 1
            else:
                term_count[term_idx] += 1
        indices.extend(term_count.keys())
        data.extend(term_count.values())
        indptr.append(len(indices))
    tf = sp.csr_matrix((data, indices, indptr), dtype=int, shape=(len(docs), len(vocab)))

    # (2) Compute normalized tfidf for each term/doc.
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(tf)
    return tfidf


def get_logger(logname):
    logger = logging.getLogger(logname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]  %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fh = logging.handlers.RotatingFileHandler(logname, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_dataset(dataset, dataset_obj):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_obj, f)


def load_dataset(dataset):
    dataset_file = TMP_DIR[dataset] + '/dataset.pkl'
    dataset_obj = pickle.load(open(dataset_file, 'rb'))
    return dataset_obj


def save_labels(dataset, labels, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    with open(label_file, 'wb') as f:
        pickle.dump(labels, f)


def load_labels(dataset, mode='train'):
    if mode == 'train':
        label_file = LABELS[dataset][0]
    elif mode == 'test':
        label_file = LABELS[dataset][1]
    else:
        raise Exception('mode should be one of {train, test}.')
    user_products = pickle.load(open(label_file, 'rb'))
    return user_products


def save_embed(dataset, embed):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    pickle.dump(embed, open(embed_file, 'wb'))


def load_embed(dataset):
    embed_file = '{}/transe_embed.pkl'.format(TMP_DIR[dataset])
    print('Load embedding:', embed_file)
    embed = pickle.load(open(embed_file, 'rb'))
    return embed


def save_kg(dataset, kg):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    pickle.dump(kg, open(kg_file, 'wb'))


def load_kg(dataset):
    kg_file = TMP_DIR[dataset] + '/kg.pkl'
    kg = pickle.load(open(kg_file, 'rb'))
    return kg

