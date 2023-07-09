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


class DataPreprocessing:
    def __init__(self, dataset_dict={},kg_emb_len = 60,st_emb_len = 60):
        
        """
        Initialise DataPreprocessing module using path to train, test, 
        user, article datasets.
        """
        
        self.st_model = None
        self.kg_model = None
        self.kg_emb_len = kg_emb_len
        self.st_emb_len = st_emb_len
        
        self.pro_train_data_df = None
        self.pro_test_data_df = None
        
        self.train_data = pd.read_csv(dataset_dict['train'])
        self.test_data = pd.read_csv(dataset_dict['test'])
        
        # Articles 
        self.df_articles = pd.read_csv(dataset_dict['articles'])
        self.df_articles['all_text'] = pd.Series(self.df_articles[['headline','teaser','text']].fillna('').values.tolist()).str.join(' ')
        
        self.df_train_test = pd.concat([self.train_data,self.test_data])
        self.tcm_clicks_count_df = self.df_train_test[self.df_train_test['response'] == 1].groupby('article_id').size().reset_index(name='tcm_click_count').sort_values( ['tcm_click_count'],ascending=False)
        self.tcm_clicks_count_df['article_popularity'] = self.tcm_clicks_count_df['tcm_click_count'] / self.tcm_clicks_count_df['tcm_click_count'].sum()
        self.articles_merged_df = pd.merge(self.df_articles, self.tcm_clicks_count_df, on=["article_id"])
        
        self.articles_merged_df = self.articles_merged_df[~self.articles_merged_df['article_id'].isin(['tcm:526-12173','tcm:526-161967','tcm:526-387709','tcm:526-635549','tcm:526-766928'])]
        self.articles_merged_df['article_length'] = self.articles_merged_df['all_text'].str.len()
        self.df_articles = self.articles_merged_df
        self.df_articles = self.df_articles[['article_id','article_popularity','all_text','article_length']]
        
        # Users
        self.df_users = pd.read_csv(dataset_dict['users'])
        self.df_users = self.df_users.loc[:, ~self.df_users.columns.isin(['f_1','xd','f_0'])]
        
        # join datasets on 'tcm_id' and 'ip'
        self.train_data = self.train_data.merge(self.df_users,on='qid')
        self.train_data = self.train_data.merge(self.df_articles,on='article_id')
        
        self.test_data = self.test_data.merge(self.df_users,on='qid')
        self.test_data = self.test_data.merge(self.df_articles,on='article_id')
        
        
        # encoding
        
        self.tcm_id_le = LabelEncoder()
        self.date_le = LabelEncoder()

        self.test_data['article_id'] = self.tcm_id_le.fit_transform(self.test_data['article_id'])
        self.train_data['article_id'] = self.tcm_id_le.transform(self.train_data['article_id'])

        unique_dates = list(self.test_data['xd'].unique()) + list(self.train_data['xd'].unique())
        self.date_le = self.date_le.fit(unique_dates)

        self.test_data['xd'] = self.date_le.transform(self.test_data['xd'])
        self.train_data['xd'] = self.date_le.transform(self.train_data['xd'])
        
        print(self.test_data.shape, self.train_data.shape)
        
    def return_inverse_transform(self,key='article_id'):
        """
        Returns initialised label encoder object to inverse transform
        encoded article tcm_id.
        """
        if key == 'article_id':
            return self.tcm_id_le
        else:
            return self.date_le
        
    def init_Transformer(self, st_name='ProsusAI/finbert'):
        """
        To use huggingface pre-trained transformer models for embeddings.
        """
        word_embedding_model = models.Transformer(st_name, max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
    def init_SentenceTransformer(self, st_name='all-MiniLM-L6-v2'):
        """
        To use pre-trained sentence transformer models for embeddings.
        """
        model = SentenceTransformer(st_name, device='cuda')
        model.max_seq_length = 512
        self.st_model = model
        
    def init_KG(self, kg_name='outputs/KGs/KG_ConceptNet_100.txt'):
        """
        Initialise with knowledge graph embedding file genrated using TucKER.
        """
        kg_embed = {}
        with open(kg_name, 'r') as kg_embed_file:
            lines = kg_embed_file.readlines()
            for line in lines:
                if line.startswith('<http://www.w3.org'):
                    continue
                if line.startswith('<http://saffron.insight'):
                    row = line.split('<http://saffron.insight-centre.org/rdf/')
                    if kg_embed.__contains__(row[1]):
                        continue
                    kg_embed[row[1].split('>')[0]] = row[1].split('>')[1].replace('\t', ' ').split()
                else:
                    row = line.replace('@en', '').split('"')
                    if kg_embed.__contains__(row[1]):
                        continue
                    if any(c.isdigit() for c in row[1]):
                        kg_embed[row[1].split()[0]] = row[1].split()[1:]
                    else:
                        kg_embed[row[1]] = row[2].split()
        kg_embed_file.close()

        self.kg_model = kg_embed
    
    def _KGE_vectorizer(self,sent):
        """
        Used to get sentence level KG embeddings.
        """
        model = self.kg_model
        sent_vec = []
        numw = 0
        for w in sent.split():
            if numw == 0:
                try:
                    vec = np.array([float(x) for x in model[w]])
                    sent_vec = vec
                except:
                    sent_vec = np.zeros(300)
            else:
                try:
                    vec = np.array([float(x) for x in model[w]])
                    sent_vec = np.add(sent_vec, vec)
                except:
                    sent_vec = sent_vec
            numw += 1
        return np.asarray(sent_vec) / numw
        
    def _STE_vectorizer(self,text,n=400):
        """
        Split texts into sentences and get embeddings for each sentence.
        The final embeddings is the mean of all sentence embeddings.
        :param text: str. Input text.
        :return: np.array. Embeddings.
        """

        text_lst = list(set(re.findall('[^!?。.？！\n]+[!?。.？！\n]?',text)))
        cnk_list = []
        cur_str = ""
        
        for i,txt in enumerate(text_lst):
            if len(cur_str.split()) >= n:
                cnk_list.append(cur_str)
                cur_str = txt
            elif i == len(text_lst) -1 :
                cur_str += txt
                cnk_list.append(cur_str)
            else:
                cur_str += txt

        return np.mean(self.st_model.encode(cnk_list),axis = 0)

    def _st_emb_vector(self, text_list):
        """
        Sentence transformer embedding, dimension reduced using UMAP.
        """
        text_list = text_list.values.tolist()
        emb_vec = np.array([self._STE_vectorizer(xi[0]) for xi in tqdm(text_list,position=0, leave=True)])
        emb_vec = umap.UMAP(n_components=self.st_emb_len, metric='cosine').fit_transform(emb_vec)
        
        return emb_vec

    def _kg_emb_vector(self, text_list):
        """
        KG embedding, dimension reduced using UMAP.
        """
        text_list = text_list.values.tolist()
        emb_vec = np.array([self._KGE_vectorizer(xi[0]) for xi in tqdm(text_list,position=0, leave=True)])
        emb_vec = umap.UMAP(n_components=self.kg_emb_len, metric='cosine').fit_transform(emb_vec)
        return emb_vec
        
        
    def fit_data_pipeline(self):
        """
        Data pre-processing pipeline defined using sklearn.pipeline.Pipeline.
        """
        text_features = ['all_text']
        numeric_features = ['f_3','f_9','f_10','f_7','f_11','article_length','article_popularity']
        categorical_features = ['xd','f_2','f_4', 'f_5', 'f_6','f_8','f_12']

        st_emb_Transformer = FunctionTransformer(self._st_emb_vector)
        kg_emb_Transformer = FunctionTransformer(self._kg_emb_vector)

        numeric_transformer = Pipeline(steps=[
               ('imputer', SimpleImputer(strategy='mean'))
              ,('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
               ('imputer', SimpleImputer(strategy='constant'))
              ,('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        preprocessor = ColumnTransformer(
           transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features),
            ('sentence_emb', st_emb_Transformer, text_features),
            ('kg_emb', kg_emb_Transformer, text_features)
        ],remainder='passthrough')
        
        try:
            self.test_data = self.test_data.drop(['Unnamed: 0'], axis=1)
            self.train_data = self.train_data.drop(['Unnamed: 0'], axis=1)
        except:
            pass
        
        self.test_data = self.test_data.sort_values(by=['qid']).reset_index(drop=True)
        self.train_data = self.train_data.sort_values(by=['qid']).reset_index(drop=True)

        
        df = preprocessor.fit_transform(self.train_data)
        # Prepare column names
        cat_columns = preprocessor.named_transformers_['categorical']['encoder'].get_feature_names(categorical_features)
        kg_columns = ['kg_'+str(i) for i in range(self.kg_emb_len)]
        st_columns = ['st_'+str(i) for i in range(self.st_emb_len)]
        rest = ['article_id', 'response', 'qid']
        
        columns = np.append(numeric_features, cat_columns)
        columns = np.append(columns, st_columns)
        columns = np.append(columns, kg_columns)
        columns = np.append(columns, rest)

        
        self.pro_train_data_df = pd.DataFrame(df, columns=columns)
        self.pro_test_data_df = pd.DataFrame(preprocessor.transform(self.test_data), columns=columns)
        
        # using dictionary to convert specific columns
        convert_dict = {'qid': int,
                        'article_id': int,
                        'response': int
                       }  

        self.pro_train_data_df = self.pro_train_data_df.astype(convert_dict)
        self.pro_test_data_df = self.pro_test_data_df.astype(convert_dict)
        
        return self.pro_train_data_df, self.pro_test_data_df
    
    def get_train_test_split(self,feature_list=[],train_data = None,test_data = None):
        """
        Split processed data into train and test according to selected feature_list. 
        """
        
        if train_data is None:
            train_data = self.pro_train_data_df
        if test_data is None:
            test_data = self.pro_test_data_df
        
        kg_columns = [ 'kg_' + str(i) for i in range(self.kg_emb_len) ]
        st_columns = [ 'st_' + str(i) for i in range(self.kg_emb_len) ]
        usr_columns = ['f_3','f_9','f_10','f_7','f_11','f_2_0','f_2_1','f_4_0','f_4_1',
                       'f_5_0','f_5_1','f_6_0','f_6_1','f_8_0','f_8_1','f_12_Aggressive Growth','f_12_Balanced','f_12_Conservative',
                       'f_12_Growth','f_12_Growth with Income','f_12_Moderate','f_12_Moderate with Income','f_12_Most Aggressive','f_12_None','f_12_Short Term']
        art_columns = ['article_length','article_popularity']

        featur_cols = []
        
        for ftr in feature_list:
            if 'art' == ftr:
                featur_cols += art_columns
            elif 'kg' == ftr:
                featur_cols += kg_columns
            elif 'usr' == ftr:
                featur_cols += usr_columns
            elif 'st' == ftr:
                featur_cols += st_columns

        print(len(featur_cols))

        X_train = train_data.loc[:, train_data.columns.isin( featur_cols )]
        y_train = train_data.loc[:, train_data.columns.isin(['response'])]
        qid_train = train_data.loc[:, train_data.columns.isin(['qid'])]
        groups_train = train_data.groupby('qid').size().to_frame('size')['size'].to_numpy()

        X_test = test_data.loc[:, test_data.columns.isin( featur_cols )]
        y_test = test_data.loc[:, test_data.columns.isin(['response'])]
        qid_test = test_data.loc[:, test_data.columns.isin(['qid'])]
        groups_test = test_data.groupby('qid').size().to_frame('size')['size'].to_numpy()

        test_data = test_data.loc[:, test_data.columns.isin( ['qid', 'article_id', 'response'] + featur_cols )]

        print(X_train.shape , y_train.shape, X_test.shape, y_test.shape)
        
        return {'train_test': [X_train, y_train, groups_train, qid_train, X_test, y_test, qid_test, groups_test], 'test_data':test_data }
        