from __future__ import absolute_import, division, print_function

import os
import numpy as np
import gzip
import pickle
from easydict import EasyDict as edict
import random

class ChallengeDataset(object):
    
    def __init__(self, data_dir, set_name='train', word_sampling_rate=1e-4):
        self.data_dir = data_dir
        if not self.data_dir.endswith('/'):
            self.data_dir +='/'
        self.train_data = set_name + '.txt'
        self.load_entities()
        self.load_article_relations()
        self.load_text()
        self.create_word_sampling_rate(word_sampling_rate)
    
    def _load_file(self, filename):
        with open(self.data_dir + filename, 'r') as f:
            return [line.strip() for line in f]
        
    def load_entities(self):
        
        entity_files = edict(
            # index of user_id 
            user='users.txt',
            # index of article_id 
            article='articles.txt',
            # index of article text word 
            word='vocab.txt',
            # index of related article
            related_article='related_article.txt',
            # topic
            topic='topics.txt',
            # product
            product='products.txt',
            # topic_tag
            topic_tag='topic_tags.txt',
            # product_tag
            product_tag='product_tags.txt'
        )
        
        for name in entity_files:
            vocab = self._load_file(entity_files[name])
            setattr(self, name, edict(vocab=vocab, vocab_size=len(vocab)))
            print('Load', name, 'of size', len(vocab))
    
    def load_text(self):
        """Load user-article reviews from train/test data files.
        Create member variable `review` associated with following attributes:
        - `data`: list of tuples (user_idx, article_idx, [word_idx...]).
        - `size`: number of reviews.
        - `article_distrib`: article vocab frequency among all eviews.
        - `article_uniform_distrib`: article vocab frequency (all 1's)
        - `word_distrib`: word vocab frequency among all reviews.
        - `word_count`: number of words (including duplicates).
        - `review_distrib`: always 1.
        """
        text_data = []  # (user_idx, article_idx, [word1_idx,...,wordn_idx])
        article_distrib = np.zeros(self.article.vocab_size)
        word_distrib = np.zeros(self.word.vocab_size)
        word_count = 0
        for line in self._load_file(self.train_data):
            arr = line.split('\t')
            user_idx = int(arr[0])
            article_idx = int(arr[1])
            word_indices = [int(i) for i in arr[2].lstrip().split(' ')]  # list of word idx
            text_data.append((user_idx, article_idx, word_indices))
            article_distrib[article_idx] += 1
            for wi in word_indices:
                word_distrib[wi] += 1
            word_count += len(word_indices)
        self.text = edict(
                data=text_data,
                size=len(text_data),
                article_distrib=article_distrib,
                article_uniform_distrib=np.ones(self.article.vocab_size),
                word_distrib=word_distrib,
                word_count=word_count,
                text_distrib=np.ones(len(text_data)) #set to 1 now
        )
        print('Load text of size', self.text.size, 'word count=', word_count)
    
    def load_article_relations(self):
        article_relations = edict(
                has_topic=('has_topic.txt', self.topic),  # (filename, entity_tail)
                has_product=('has_product.txt', self.product),
                also_response=('also_response.txt', self.related_article),
                recommended_together=('recommended_together.txt', self.related_article),
                response_together=('response_together.txt', self.related_article),
                has_topic_tag=('has_topic_tag.txt',self.topic_tag),
                has_product_tag=('has_product_tag.txt',self.product_tag)
        )
        for name in article_relations:
            # We save information of entity_tail (et) in each relation.
            # Note that `data` variable saves list of entity_tail indices.
            # The i-th record of `data` variable is the entity_tail idx (i.e. product_idx=i).
            # So for each product-relation, there are always |products| records.
            relation = edict(
                    data=[],
                    et_vocab=article_relations[name][1].vocab, #copy of brand, catgory ... 's vocab 
                    et_distrib=np.zeros(article_relations[name][1].vocab_size) #[1] means self.brand ..
            )
            for line in self._load_file(article_relations[name][0]): #[0] means brand_p_b.txt.gz ..
                knowledge = []
                for x in line.split(' '):  # some lines may be empty
                    if len(x) > 0:
                        x = int(x)
                        knowledge.append(x)
                        relation.et_distrib[x] += 1
                relation.data.append(knowledge)
            setattr(self, name, relation)
            print('Load', name, 'of size', len(relation.data))
            
    def create_word_sampling_rate(self, sampling_threshold):
        print('Create word sampling rate')
        self.word_sampling_rate = np.ones(self.word.vocab_size)
        if sampling_threshold <= 0:
            return
        threshold = sum(self.text.word_distrib) * sampling_threshold
        for i in range(self.word.vocab_size):
            if self.text.word_distrib[i] == 0:
                continue
            self.word_sampling_rate[i] = min((np.sqrt(float(self.text.word_distrib[i]) / threshold) + 1) * threshold / float(self.text.word_distrib[i]), 1.0)

class ChallengeDataLoader(object):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.text_size = self.dataset.text.size
        self.article_relations = ['has_product', 'has_topic','has_topic_tag','has_product_tag','also_response','recommended_together', 'response_together']
        self.finished_word_num = 0
        self.reset()
    
    def reset(self):
        # Shuffle text order
        self.text_seq = np.random.permutation(self.text_size)
        self.cur_text_i = 0
        self.cur_word_i = 0
        self._has_next = True
    
    def get_batch(self):
        """Return a matrix of [batch_size x 8], where each row contains
        (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        batch = []
        text_idx = self.text_seq[self.cur_text_i]
        user_idx, article_idx, text_list = self.dataset.text.data[text_idx]
        article_knowledge = {pr: getattr(self.dataset, pr).data[article_idx] for pr in self.article_relations}
        while len(batch) < self.batch_size:
            # 1) Sample the word
            word_idx = text_list[self.cur_word_i]
            if random.random() < self.dataset.word_sampling_rate[word_idx]:
                data = [user_idx, article_idx, word_idx]
                for pr in self.article_relations:
                    if len(article_knowledge[pr]) <= 0:
                        data.append(-1)
                    else:
                        data.append(random.choice(article_knowledge[pr]))
                batch.append(data)

            # 2) Move to next word/text
            self.cur_word_i += 1
            self.finished_word_num += 1
            if self.cur_word_i >= len(text_list):
                self.cur_text_i += 1
                if self.cur_text_i >= self.text_size:
                    self._has_next = False
                    break
                self.cur_word_i = 0
                text_idx = self.text_seq[self.cur_text_i]
                user_idx, article_idx, text_list = self.dataset.text.data[text_idx]
                article_knowledge = {pr: getattr(self.dataset, pr).data[article_idx] for pr in self.article_relations}

        return np.array(batch)

    def has_next(self):
        """Has next batch."""
        return self._has_next