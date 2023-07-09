from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
from math import log
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import gzip
import pickle
import random
from datetime import datetime
import matplotlib.pyplot as plt
import torch

from utils import *
from data_utils import ChallengeDataset


class KnowledgeGraph(object):

    def __init__(self, dataset):
        self.G = dict()
        self._load_entities(dataset)
        self._load_text(dataset)
        self._load_knowledge(dataset)
        self._clean()
        self.top_matches = None

    def _load_entities(self, dataset):
        print('Load entities...')
        num_nodes = 0
        for entity in get_entities():
            self.G[entity] = {}
            vocab_size = getattr(dataset, entity).vocab_size
            for eid in range(vocab_size):
                self.G[entity][eid] = {r: [] for r in get_relations(entity)}
            num_nodes += vocab_size
        print('Total {:d} nodes.'.format(num_nodes))

    def _load_text(self, dataset, word_tfidf_threshold=0.1, word_freq_threshold=5000):
        print('Load text...')
        # (1) Filter words by both tfidf and frequency.
        vocab = dataset.word.vocab
        text = [d[2] for d in dataset.text.data]
        text_tfidf = compute_tfidf_fast(vocab, text)
        distrib = dataset.text.word_distrib

        num_edges = 0
        all_removed_words = []
        for rid, data in enumerate(dataset.text.data):
            uid, aid, text = data
            doc_tfidf = text_tfidf[rid].toarray()[0]
            '''
            for wid in set(text):
                if distrib[wid] <= word_freq_threshold:
                    print(doc_tfidf[wid],distrib[wid])
                if doc_tfidf[wid] >= word_tfidf_threshold and distrib[wid] <= word_freq_threshold:
                    print(wid)
            '''
            remained_words = [wid for wid in set(text)
                              if doc_tfidf[wid] >= word_tfidf_threshold
                              and distrib[wid] <= word_freq_threshold]
            removed_words = set(text).difference(remained_words)  # only for visualize
            removed_words = [vocab[wid] for wid in removed_words]
            all_removed_words.append(removed_words)
            if len(remained_words) <= 0:
                continue

            # (2) Add edges.
            self._add_edge(USER, uid, RECOMMENDED, ARTICLE, aid)
            num_edges += 2
            for wid in remained_words:
                self._add_edge(ARTICLE, aid, WITHIN, WORD, wid)
                num_edges += 2
        print('Total {:d} text edges.'.format(num_edges))

        with open('./tmp/text_removed_words.txt', 'w') as f:
            f.writelines([' '.join(words) + '\n' for words in all_removed_words])

    def _load_knowledge(self, dataset):
        for relation in [HAS_TOPIC,HAS_PRODUCT, ALSO_RESPONSE, RECOMMENDED_TOGETHER, RESPONSE_TOGETHER, HAS_TOPIC_TAG, HAS_PRODUCT_TAG]:
            print('Load knowledge {}...'.format(relation))
            data = getattr(dataset, relation).data
            num_edges = 0
            for aid, eids in enumerate(data):
                if len(eids) <= 0:
                    continue
                for eid in set(eids):
                    et_type = get_entity_tail(ARTICLE, relation)
                    self._add_edge(ARTICLE, aid, relation, et_type, eid)
                    num_edges += 2
            print('Total {:d} {:s} edges.'.format(num_edges, relation))

    def _add_edge(self, etype1, eid1, relation, etype2, eid2):
        self.G[etype1][eid1][relation].append(eid2)
        self.G[etype2][eid2][relation].append(eid1)

    def _clean(self):
        print('Remove duplicates...')
        for etype in self.G:
            for eid in self.G[etype]:
                for r in self.G[etype][eid]:
                    data = self.G[etype][eid][r]
                    data = tuple(sorted(set(data)))
                    self.G[etype][eid][r] = data

    def compute_degrees(self):
        print('Compute node degrees...')
        self.degrees = {}
        self.max_degree = {}
        for etype in self.G:
            self.degrees[etype] = {}
            for eid in self.G[etype]:
                count = 0
                for r in self.G[etype][eid]:
                    count += len(self.G[etype][eid][r])
                self.degrees[etype][eid] = count

    def get(self, eh_type, eh_id=None, relation=None):
        data = self.G
        if eh_type is not None:
            data = data[eh_type]
        if eh_id is not None:
            data = data[eh_id]
        if relation is not None:
            data = data[relation]
        return data

    def __call__(self, eh_type, eh_id=None, relation=None):
        return self.get(eh_type, eh_id, relation)

    def get_tails(self, entity_type, entity_id, relation):
        return self.G[entity_type][entity_id][relation]

    def get_tails_given_user(self, entity_type, entity_id, relation, user_id):
        """ Very important!
        :param entity_type:
        :param entity_id:
        :param relation:
        :param user_id:
        :return:
        """
        tail_type = KG_RELATION[entity_type][relation]
        tail_ids = self.G[entity_type][entity_id][relation]
        if tail_type not in self.top_matches:
            return tail_ids
        top_match_set = set(self.top_matches[tail_type][user_id])
        top_k = len(top_match_set)
        if len(tail_ids) > top_k:
            tail_ids = top_match_set.intersection(tail_ids)
        return list(tail_ids)

    def trim_edges(self):
        degrees = {}
        for entity in self.G:
            degrees[entity] = {}
            for eid in self.G[entity]:
                for r in self.G[entity][eid]:
                    if r not in degrees[entity]:
                        degrees[entity][r] = []
                    degrees[entity][r].append(len(self.G[entity][eid][r]))

        for entity in degrees:
            for r in degrees[entity]:
                tmp = sorted(degrees[entity][r], reverse=True)
                print(entity, r, tmp[:10])

    def set_top_matches(self, u_u_match, u_a_match, u_w_match):
        self.top_matches = {
            USER: u_u_match,
            ARTICLE: u_a_match,
            WORD: u_w_match,
        }
    
    def heuristic_search(self, uid, aid, pattern_id, trim_edges=False):
        if trim_edges and self.top_matches is None:
            raise Exception('To enable edge-trimming, must set top_matches of users first!')
        if trim_edges:
            _get = lambda e, i, r: self.get_tails_given_user(e, i, r, uid)
        else:
            _get = lambda e, i, r: self.get_tails(e, i, r)

        pattern = PATH_PATTERN[pattern_id]
        paths = []
        '''
        if pattern_id == 1:
            aids_u = set(_get(USER, uid, RECOMMENDED))  # USER->RECOMMENDED->ARTICLE
            aids_a = set(_get(ARTICLE, aid, RESPONSE))  # ARTICLE->HAS->WORD
            
            intersect_nodes = wids_u.intersection(wids_a)
            paths = [(uid, x, aid) for x in intersect_nodes]
        if pattern_id == 1:  # OK
            wids_u = set(_get(USER, uid, RESPONSE))  # USER->RESPONSE->ARTICLE
            wids_a = set(_get(ARTICLE, aid, WITHIN))  # ARTICLE->HAS->WORD
            intersect_nodes = wids_u.intersection(wids_a)
            paths = [(uid, x, aid) for x in intersect_nodes]
        '''
        '''
        if pattern_id == 1:
            aids_u = set(_get(USER, uid, RECOMMENDED))  # USER->RECOMMENDED->ARTICLE
            aids_a = set(_get(ARTICLE, aid, RESPONSE))  # ARTICLE->RESPONSE->USER
            intersect_nodes = aids_u.intersection(aids_a)
            paths = [(uid, x, aid) for x in intersect_nodes]
        elif pattern_id in [11, 12, 13, 14, 15, 16, 17, 19, 20]:
        '''
        if pattern_id in [11, 13, 14, 15, 16, 17, 19, 20]:
            aids_u = set(_get(USER, uid, RECOMMENDED))  # USER->RECOMMENDED->ARTICLE
            aids_u = aids_u.difference([aid])  # exclude target product
            nodes_p = set(_get(ARTICLE, aid, pattern[3][0]))  # PRODUCT->relation->node2
            if pattern[2][1] == USER:
                nodes_p.difference([uid])
            for aid_u in aids_u:
                relation, entity_tail = pattern[2][0], pattern[2][1]
                et_ids = set(_get(ARTICLE, aid_u, relation))  # USER->PURCHASE->PRODUCT->relation->node2
                intersect_nodes = et_ids.intersection(nodes_p)
                tmp_paths = [(uid, aid_u, x, aid) for x in intersect_nodes]
                paths.extend(tmp_paths)
        elif pattern_id == 18:
            #18: ((None, USER), (RECOMMENDED, ARTICLE),(RECOMMENDED, USER),(RECOMMENDED, ARTICLE))
            uids_u = set(_get(USER, uid, RECOMMENDED))  # USER->RECOMMENDED->ARTICLE
            uids_a = set(_get(ARTICLE, aid, RECOMMENDED))  # ARTICLE->RECOMMENDED->USER
            uids_a = uids_p.difference([uid])  # exclude source user
            for uid_a in uids_a:
                #ARTICLE->RECOMMENDED->USER->RECOMMENDED->ARTICLE
                uids_u_a = set(_get(USER, uid_a, RECOMMENDED))  # PRODUCT->PURCHASE->USER->MENTION->WORD
                intersect_nodes = uids_u.intersection(uids_u_a)
                tmp_paths = [(uid, x, uid_a, aid) for x in intersect_nodes]
                paths.extend(tmp_paths)

        return paths


def check_test_path(dataset_str, kg):
    # Check if there exists at least one path for any user-product in test set.
    test_user_products = load_labels(dataset_str, 'test')
    for uid in test_user_products:
        for pid in test_user_products[uid]:
            count = 0
            #for pattern_id in [1, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            for pattern_id in [11, 13, 14, 15, 16, 17, 18, 19, 20]:
                tmp_path = kg.heuristic_search(uid, pid, pattern_id)
                count += len(tmp_path)
            if count == 0:
                print(uid, pid)

