import pandas as pd
import numpy as np
import re
import xgboost as xgb
import os
import pickle

def load_all():
    train_df = pd.read_csv('recsys_data/train.csv')
    articles_df = pd.read_csv('recsys_data/articles.csv')
    users_df = pd.read_csv('recsys_data/users.csv')
    test_df = pd.read_csv('recsys_data/test.csv')
    results_df = pd.read_csv('recsys_data/results.csv')
    return train_df, articles_df, users_df, test_df, results_df


def get_embeddings(model,text):
    """
    Split texts into sentences and get embeddings for each sentence.
    The final embeddings is the mean of all sentence embeddings.
    :param text: str. Input text.
    :return: np.array. Embeddings.
    """
    return np.mean(
        model.encode(
            list(set(re.findall('[^!?。.？！]+[!?。.？！]?', text)))
        ), axis=0)


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def eval_ranker(actual,predicted,k=10):
    #Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []

    for a,p in zip(actual, predicted):
        if len(p)>k:
            p = p[:k]
        rel_set, pred_list = a,p
        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (np.log(i +2) / np.log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (np.log(i + 2)/np.log(2))
        ndcg = dcg / idcg
        recall = hit_num / len(rel_set)
        precision = hit_num / len(pred_list)
        hit = 1.0 if hit_num > 0.0 else 0.0


        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)


    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_ndcg = np.mean(ndcgs)
    avg_hit = np.mean(hits)

    print('NDCG={:.3f} | Recall={:.3f} | Precision={:.3f} '.format(avg_ndcg,avg_recall,avg_hit,avg_precision))
    return {'NDCG' : avg_ndcg, 'Recall' : avg_recall, 'Precision' : avg_precision}

def model_save(model,path,DataPrep):
    if not os.path.exists(path):
        os.makedirs(path)
    model.save_model(path+'model.json')
    with open(path+'dataprep.pkl','wb') as f:
        pickle.dump(DataPrep, f)


def load_model(path):
    model = xgb.XGBRanker()
    model.load_model(path+'model.json')
    with open(path+'dataprep.pkl','rb') as f:
        DataPrep = pickle.load(f)
    return model , DataPrep
