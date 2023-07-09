import numpy as np                                # For matrix operations and numerical processing
import pandas as pd                               # For munging tabular data
from sentence_transformers import SentenceTransformer, util,models
import umap
import re
import os 
from sklearn.preprocessing import LabelEncoder,StandardScaler, OrdinalEncoder,FunctionTransformer,OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import recmetrics
import ltr_utils as ut



class LtrPrediction:
    def __init__(self, xgb_model, test_data):
        """
        Initialise LtrPrediction with xgb_model and test data.
        Performed group prediction using the model to get the prefrence value.
        """
        self.model = xgb_model
        self.test_data = test_data
        
        group_prediction = test_data.groupby('qid').apply(self._group_predict).reset_index()
        self.test_data_ = test_data.loc[:, ['article_id', 'response', 'qid']]
        self.test_data_['response_pred'] = group_prediction.explode(0)[0].values
        
        self.test_data_.response_pred = pd.to_numeric(self.test_data_.response_pred, errors='coerce')

        pivot_pred = self.test_data_.pivot_table(index='qid', columns='article_id', values='response_pred').fillna(0)
        
        self.test_data_actual_ids = self.test_data_[self.test_data_['response'] == 1].copy().groupby('qid', as_index=False)['article_id'].agg({'article_actual': (lambda x: list(set(x)))})
        self.test_data_actual_ids = self.test_data_actual_ids.set_index("qid")
        # make recommendations for all members in the test data
        self.article_predictions_list = []
        for ip in self.test_data_actual_ids.index:
            article_predictions = self._get_users_predictions(ip, pivot_pred, 10)
            self.article_predictions_list.append(article_predictions)

        self.test_data_actual_prediction_ids = self.test_data_actual_ids.copy()
        self.test_data_actual_prediction_ids['article_prediction'] = self.article_predictions_list
        
    def predict(self):
        """
        Returns dataframe vwith actual and predicted tcm.id
        """
        return self.test_data_
        
    def _get_users_predictions(self,ip, pivot_pred,k=10):
        """
        To get prdiction/recommendation per user.
        """
        recommended_items = pd.DataFrame(pivot_pred.loc[ip])
        recommended_items.columns = ["predicted_"]
        recommended_items = recommended_items.sort_values('predicted_', ascending=False)    
        recommended_items = recommended_items.head(k)
        return recommended_items.index.tolist()

    def _group_predict(self,grp_rows):
        """ 
        Return size value for the max weight line 
        """
        return  list(self.model.predict(grp_rows.loc[:, ~grp_rows.columns.isin(['article_id', 'response', 'qid'])]))
    
    def evaluate(self,K=10):
        """
        Evaluate recommended article using recmetrics.
        """
        article_actuals = self.test_data_actual_prediction_ids.article_actual.values.tolist()
        article_predictions = self.test_data_actual_prediction_ids.article_prediction.values.tolist()

        results_dict = {}
        #Prediction MAP
        results_dict['MAP@'+str(K)] = ut.mapk(article_actuals, article_predictions, k=K)
        results_dict['results@'+str(K)] = ut.eval_ranker(article_actuals, article_predictions, k=K)
        
        
        return results_dict
    
    def get_recomendation(self):
        """
        Returns dataframe with per user average precision score.
        """
        data = self.test_data_actual_prediction_ids
        data['apk'] = data.apply(lambda row: ut.apk(row['article_actual'],row['article_prediction']) ,axis=1)
        
        return data
