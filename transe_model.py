from __future__ import absolute_import, division, print_function

from easydict import EasyDict as edict
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from data_utils import ChallengeDataset


class KnowledgeEmbedding(nn.Module):
    def __init__(self, dataset, device='cuda',seed=123,gpu=1,epochs=10, batch_size=16,lr=0.5,weight_decay=0,l2_lambda=0,max_grad_norm=5.0,embed_size=2,num_neg_samples=5,steps_per_checkpoint=200):
        super(KnowledgeEmbedding, self).__init__()
        self.embed_size = embed_size
        self.num_neg_samples = num_neg_samples
        self.device = device
        self.l2_lambda = l2_lambda

        # Initialize entity embeddings.
        self.entities = edict(
            user=edict(vocab_size=dataset.user.vocab_size),
            article=edict(vocab_size=dataset.article.vocab_size),
            word=edict(vocab_size=dataset.word.vocab_size),
            related_article=edict(vocab_size=dataset.related_article.vocab_size),
            topic=edict(vocab_size=dataset.topic.vocab_size),
            product=edict(vocab_size=dataset.product.vocab_size),
            topic_tag=edict(vocab_size=dataset.topic_tag.vocab_size),
            product_tag=edict(vocab_size=dataset.product_tag.vocab_size)
        )
        for e in self.entities:
            embed = self._entity_embedding(self.entities[e].vocab_size)
            setattr(self, e, embed)

        # Initialize relation embeddings and relation biases.
        self.relations = edict(
            # this might be has some problem
            response=edict(
                et='article',
                #article_uniform_distrib?
                et_distrib=self._make_distrib(dataset.text.article_uniform_distrib)),
            recommended=edict(
                et='article',
                et_distrib=self._make_distrib(dataset.text.article_distrib)),
            within=edict(
                et='word',
                et_distrib=self._make_distrib(dataset.text.word_distrib)),
            has_topic=edict(
                et='topic',
                et_distrib=self._make_distrib(dataset.has_topic.et_distrib)),
            has_product=edict(
                et='product',
                et_distrib=self._make_distrib(dataset.has_product.et_distrib)),
            has_topic_tag=edict(
                et='topic_tag',
                et_distrib=self._make_distrib(dataset.has_topic_tag.et_distrib)),
            has_product_tag=edict(
                et='product_tag',
                et_distrib=self._make_distrib(dataset.has_product_tag.et_distrib)),
            also_response=edict(
                et='related_article',
                et_distrib=self._make_distrib(dataset.also_response.et_distrib)),
            recommended_together=edict(
                et='related_article',
                et_distrib=self._make_distrib(dataset.recommended_together.et_distrib)),
            response_together=edict(
                et='related_article',
                et_distrib=self._make_distrib(dataset.response_together.et_distrib)),
        )
        for r in self.relations:
            embed = self._relation_embedding()
            setattr(self, r, embed)
            bias = self._relation_bias(len(self.relations[r].et_distrib))
            setattr(self, r + '_bias', bias)

    def _entity_embedding(self, vocab_size):
        """Create entity embedding of size [vocab_size+1, embed_size].
            Note that last dimension is always 0's.
        """
        embed = nn.Embedding(vocab_size + 1, self.embed_size, padding_idx=-1, sparse=False)
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(vocab_size + 1, self.embed_size).uniform_(-initrange, initrange)
        embed.weight = nn.Parameter(weight)
        return embed

    def _relation_embedding(self):
        """Create relation vector of size [1, embed_size]."""
        initrange = 0.5 / self.embed_size
        weight = torch.FloatTensor(1, self.embed_size).uniform_(-initrange, initrange)
        embed = nn.Parameter(weight)
        return embed

    def _relation_bias(self, vocab_size):
        """Create relation bias of size [vocab_size+1]."""
        bias = nn.Embedding(vocab_size + 1, 1, padding_idx=-1, sparse=False)
        bias.weight = nn.Parameter(torch.zeros(vocab_size + 1, 1))
        return bias

    def _make_distrib(self, distrib):
        """Normalize input numpy vector to distribution."""
        distrib = np.power(np.array(distrib, dtype=np.float), 0.75)
        distrib = distrib / distrib.sum()
        distrib = torch.FloatTensor(distrib).to(self.device)
        return distrib

    def forward(self, batch_idxs):
        loss = self.compute_loss(batch_idxs)
        return loss

    def compute_loss(self, batch_idxs):
        """Compute knowledge graph negative sampling loss.
        batch_idxs: batch_size * 8 array, where each row is
                (u_id, p_id, w_id, b_id, c_id, rp_id, rp_id, rp_id).
        """
        user_idxs = batch_idxs[:, 0]
        article_idxs = batch_idxs[:, 1]
        word_idxs = batch_idxs[:, 2]
        product_idxs = batch_idxs[:, 3]
        topic_idxs = batch_idxs[:, 4]
        topic_tag_idxs = batch_idxs[:, 5]
        product_tag_idxs = batch_idxs[:, 6]
        rarticle1_idxs = batch_idxs[:, 7]
        rarticle2_idxs = batch_idxs[:, 8]
        rarticle3_idxs = batch_idxs[:, 9]

        regularizations = []

        # user + response -> article
        ua_loss, ua_embeds = self.neg_loss('user', 'response', 'article', user_idxs, article_idxs)
        regularizations.extend(ua_embeds)
        loss = ua_loss

        # user + recommended -> article
        ur_loss, ur_embeds = self.neg_loss('user', 'recommended', 'article', user_idxs, article_idxs)
        regularizations.extend(ur_embeds)
        loss += ur_loss

        # article + within -> word
        pw_loss, pw_embeds = self.neg_loss('article', 'within', 'word', article_idxs, word_idxs)
        regularizations.extend(pw_embeds)
        loss += pw_loss

        # article + has_topic -> topic
        pb_loss, pb_embeds = self.neg_loss('article', 'has_topic', 'topic', article_idxs, topic_idxs)
        if pb_loss is not None:
            regularizations.extend(pb_embeds)
            loss += pb_loss

        # article + has_product -> product
        pc_loss, pc_embeds = self.neg_loss('article', 'has_product', 'product', article_idxs, product_idxs)
        if pc_loss is not None:
            regularizations.extend(pc_embeds)
            loss += pc_loss
            
        # article + has_topic_tag -> topic_tag
        pc_loss, pc_embeds = self.neg_loss('article', 'has_topic_tag', 'topic_tag', article_idxs, topic_tag_idxs)
        if pc_loss is not None:
            regularizations.extend(pc_embeds)
            loss += pc_loss
            
        # article + has_product_tag -> product_tag
        pc_loss, pc_embeds = self.neg_loss('article', 'has_product_tag', 'product_tag', article_idxs, product_tag_idxs)
        if pc_loss is not None:
            regularizations.extend(pc_embeds)
            loss += pc_loss

        # article + also_bought -> related_article1
        ar1_loss, ar1_embeds = self.neg_loss('article', 'also_response', 'related_article', article_idxs, rarticle1_idxs)
        if ar1_loss is not None:
            regularizations.extend(ar1_embeds)
            loss += ar1_loss

        # article + recommended_together -> related_article2
        ar2_loss, ar2_embeds = self.neg_loss('article', 'recommended_together', 'related_article', article_idxs, rarticle2_idxs)
        if ar2_loss is not None:
            regularizations.extend(ar2_embeds)
            loss += ar2_loss

        # article + bought_together -> related_article3
        ar3_loss, ar3_embeds = self.neg_loss('article', 'response_together', 'related_article', article_idxs, rarticle3_idxs)
        if ar3_loss is not None:
            regularizations.extend(ar3_embeds)
            loss += ar3_loss

        # l2 regularization
        if self.l2_lambda > 0:
            l2_loss = 0.0
            for term in regularizations:
                l2_loss += torch.norm(term)
            loss += self.l2_lambda * l2_loss

        return loss

    def neg_loss(self, entity_head, relation, entity_tail, entity_head_idxs, entity_tail_idxs):
        # Entity tail indices can be -1. Remove these indices. Batch size may be changed!
        mask = entity_tail_idxs >= 0
        fixed_entity_head_idxs = entity_head_idxs[mask]
        fixed_entity_tail_idxs = entity_tail_idxs[mask]
        if fixed_entity_head_idxs.size(0) <= 0:
            return None, []
        entity_head_embedding = getattr(self, entity_head)  # nn.Embedding
        entity_tail_embedding = getattr(self, entity_tail)  # nn.Embedding
        relation_vec = getattr(self, relation)  # [1, embed_size]
        relation_bias_embedding = getattr(self, relation + '_bias')  # nn.Embedding
        entity_tail_distrib = self.relations[relation].et_distrib  # [vocab_size]

        return kg_neg_loss(entity_head_embedding, entity_tail_embedding,
                           fixed_entity_head_idxs, fixed_entity_tail_idxs,
                           relation_vec, relation_bias_embedding, self.num_neg_samples, entity_tail_distrib)


def kg_neg_loss(entity_head_embed, entity_tail_embed, entity_head_idxs, entity_tail_idxs,
                relation_vec, relation_bias_embed, num_samples, distrib):
    """Compute negative sampling loss for triple (entity_head, relation, entity_tail).

    Args:
        entity_head_embed: Tensor of size [batch_size, embed_size].
        entity_tail_embed: Tensor of size [batch_size, embed_size].
        entity_head_idxs:
        entity_tail_idxs:
        relation_vec: Parameter of size [1, embed_size].
        relation_bias: Tensor of size [batch_size]
        num_samples: An integer.
        distrib: Tensor of size [vocab_size].

    Returns:
        A tensor of [1].
    """
    batch_size = entity_head_idxs.size(0)
    entity_head_vec = entity_head_embed(entity_head_idxs)  # [batch_size, embed_size]
    example_vec = entity_head_vec + relation_vec  # [batch_size, embed_size]
    example_vec = example_vec.unsqueeze(2)  # [batch_size, embed_size, 1]

    entity_tail_vec = entity_tail_embed(entity_tail_idxs)  # [batch_size, embed_size]
    pos_vec = entity_tail_vec.unsqueeze(1)  # [batch_size, 1, embed_size]
    relation_bias = relation_bias_embed(entity_tail_idxs).squeeze(1)  # [batch_size]
    pos_logits = torch.bmm(pos_vec, example_vec).squeeze() + relation_bias  # [batch_size]
    pos_loss = -pos_logits.sigmoid().log()  # [batch_size]

    neg_sample_idx = torch.multinomial(distrib, num_samples, replacement=True).view(-1)
    neg_vec = entity_tail_embed(neg_sample_idx)  # [num_samples, embed_size]
    neg_logits = torch.mm(example_vec.squeeze(2), neg_vec.transpose(1, 0).contiguous())
    neg_logits += relation_bias.unsqueeze(1)  # [batch_size, num_samples]
    neg_loss = -neg_logits.neg().sigmoid().log().sum(1)  # [batch_size]

    loss = (pos_loss + neg_loss).mean()
    return loss, [entity_head_vec, entity_tail_vec, neg_vec]

