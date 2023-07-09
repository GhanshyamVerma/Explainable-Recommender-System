
# KG Driven XGBoost based Recomendation System
Recommendatiion System is built using learning to rank approach. Popular boosting
models like XGBoostRanker, LightGBMRanker, CatBoostRanker trained on Recsys Challenge data incorporating user article, features, KG embeddings,SentenceTransformer embeddings.


## Table of contents


* [Installation]()
* [Top-level directory layout]()
* [Usage]()
    * [Knowledge Graph Embeddings]()
    * [XGBoost_model]()
        * [Data Sampling]()
        * [Data Preprocessing]()
        * [Model Training]()
        * [Hyperparameter Tuning]()
        * [Model Evaluation]()


## Installation

Install project dependencies 

```bash
  pip install -r requirements.txt
```
TuckER for Knowledge Graph embeddings 

```bash
 git clone https://github.com/ibalazevic/TuckER.git
```    
ml_metrics is the Python implementation of Metrics implementations a library of various supervised machine learning evaluation metrics.
```bash
 git clone https://github.com/benhamner/Metrics.git
 cd Metrics/Python
 python setup.py install
``` 
recmetrics - A python library of evalulation metrics and diagnostic tools for recommender systems.
```bash
 pip install recmetrics
```

## Top-level directory layout


    .
    ├── ...
    ├── recsys                      # Project Directory
    │   ├── KG_nt                   # Folder containing  Saffron generated KGs
    │   ├── outputs                 # Contains processed datasets, program outputs
    │   ├── recmetrics              # RecSys evaluation package
    │   ├── recsys_data             # RecSys challenge data
    │   ├── TuckER                  # KG embedding package
    │   ├── prediction.py           # Custom module for LTR prediction evaluation
    │   ├── processing.py           # Custom module for data preparation
    │   ├── ltr_utils.py            # Custom utility functions
    │   ├── data_explore.ipynb      # RecSys data exploration
    │   ├── process_kg.py           # Pre-process data for TucKER 
    │   ├── XGBoost_model.ipynb     # Train hyper-tune evaluate XGBoost Ranker
    │   ├── LGBM_model.ipynb        # Train hyper-tune evaluate LightGBM Ranker
    │   ├── CatBoost_model.ipynb    # Train hyper-tune evaluate CatBoost Ranker
    │   ├── requirements.txt        # Project dependency
    │   └── README.md               # Project description
    └── ...



## Usage


### Knowledge Graph Embeddings

Saffron was used to generate knowledge graph from unstructred data (article content). TucKER was used on KG .nt files to generate the embeddings.

Go to the project directory

```bash
  cd resys
```

Split Knowledge Graph .nt file into train,test and validation to train TucKER .

```bash
  python process_kg.py --kg_path ../KG_nt/KG_dep_parsing_100terms.nt --out_path data/KG_dep_parsing_100terms

```
TuckER module needs some aditional code to save the kg embeddings in a text file . Below code is needed to be apended at the end of  train_and_eval() function definition which can be found inside main.py .

```bash
data_idxs = self.get_data_idxs(d.data)
er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
print("Number of data points: %d" % len(data_idxs))

entity_embeddings = {}

for i in range(len(d.entities)):
    entity_emb = model.E(torch.tensor([self.entity_idxs[d.entities[i]]]).cuda())
    entity_embeddings[d.entities[i]] = entity_emb

relation_embeddings = {}
for i in range(len(d.relations)):
    relation_emb = model.R(torch.tensor([self.relation_idxs[d.relations[i]]]).cuda())
    relation_embeddings[d.relations[i]] = relation_emb

f = open("opendialkg_entity_embeddings.txt", "w", encoding="utf-8")
for key in entity_embeddings:
    l = str(key)
    emb = entity_embeddings[key].cpu().detach().numpy()
    for i in range(emb.shape[1]):
        l += "\t" + str(emb[0][i])
    l += "\n"
    f.write(l)
f.close()
f = open("opendialkg_relation_embeddings.txt", "w", encoding="utf-8")
for key in relation_embeddings:
    l = str(key)
    emb = relation_embeddings[key].cpu().detach().numpy()
    for i in range(emb.shape[1]):
        l += "\t" + str(emb[0][i])
    l += "\n"
    f.write(l)
f.close()

```
Run TuckER

```bash
  cd TuckER
  CUDA_VISIBLE_DEVICES=0 python main.py --dataset KG_dep_parsing_500terms --num_iterations 300 --batch_size 128 --lr 0.05 --dr 1.0 --edim 300 --rdim 300 --input_dropout 0.3 --hidden_dropout1 0.4 --hidden_dropout2 0.5 --label_smoothing 0.1
  mv opendialkg_entity_embeddings.txt outputs/KGs/KG_dep_parsing_500terms.txt

```
### TransE Embeddings

The `TransE_embedding_creation.ipynb` file contains the steps for generating TransE embeddings created from structured Knowledge Graphs. The TransE embedding pickle file is created by running the `KG_RL_Rec_Sys.ipynb` file in root directory of the repository. The map files containing the maping of each word, topic, product, etc. with its corresponding indices is generated from the `processing.ipynb` in the root direcotory of the repository. Using the pickle file containing the TransE embeddings and the map files with the mapping for each embedding, a combined transE embedding file can be created. To avoid duplicates, the combined embedding is created by adding all the embeddings for words then other unique products, product tags, topic and topic tags are added.  

### XGBoost_model

Multiple Ranking models like XGBoost, LightGBM and CatBoost were implemented on RecSys data incorporating KG and sentence tarnsformer embeddings. XGBoost_model.ipynb, LightGBM_model.ipynb and CatBoost_model.ipynb follow same code structure. Here, XGBoost_model.ipynb code sections are explained in details.

#### Data Sampling

The train and test dataset contains large amount of customer and article interaction data through clicks. Thus, the model tarin time was significantly long. To address the issue mulitple samples were drawn using different strategies to train the models. The sampling technique mentioned below selectes the top 100 customers with highest number of clicks.

```bash
dv_train = pd.read_csv('recsys_data/train.csv')
ips = dv_train[dv_train['response'] == 1].groupby('ip').size().reset_index(name='num_clicks').sort_values(by=['num_clicks'], ascending=False)['ip'].head(100)
df_ips = pd.DataFrame({'ip':ips})
dv_train = pd.merge(dv_train,df_ips,on='ip')

dv_test = pd.read_csv('outputs/test_all.csv')
dv_test = pd.merge(dv_test,df_ips,on='ip')

```
#### Data Preprocessing

From the processing module, the customed DataPreprocessing class is used to perform all the required data preprocessing operations. A pipeline is created using sklearn.pipeline.Pipeline to process and merge KG embeddings and Sentence Tarnsformer embeddings to generate the processed training test file for model training.

```bash
from processing import DataPreprocessing

DataPrep = DataPreprocessing(data_dict)
DataPrep.init_KG('outputs/KGs/KG_dep_parsing_100.txt')
DataPrep.init_SentenceTransformer('all-MiniLM-L6-v2')
train_data_df, test_data_df = DataPrep.fit_data_pipeline()

```


#### Model Training

```bash

xgb_params = {  'booster':"gbtree", 
                'objective':"rank:pairwise",
                'tree_method':"gpu_hist", 
                'sampling_method':"gradient_based",
                'eval_metric':['map@10'],
             }

model = xgb.XGBRanker(**xgb_params)


model.fit(X_train, y_train, group=groups_train, eval_set=[(X_train, y_train),(X_test, y_test)], eval_group=[groups_train,groups_test],  verbose=False)

```

#### Hyperparameter Tuning

Optuna is used to find the optimum hyperparameter for model training. Optuna is a software framework for automating the optimization process of these hyperparameters. It automatically finds optimal hyperparameter values by making use of different samplers such as grid search, random, bayesian, and evolutionary algorithms. For the project,MAP@10 metric is maximised to find the optimised hyperparameters.

```bash

metric = "map@10"

tuning_params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8, step=1),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8, step=0.05),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, step=0.01),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.01, 10.0, step=0.01),
            "gamma": trial.suggest_float("gamma", 0.01, 10.0, step=0.01),
            "max_delta_step":  trial.suggest_float("max_delta_step", 0.01, 10.0, step=0.01),
            "eta": trial.suggest_float("eta", 0.001, 10.0, step=0.001)

        }

```


#### Model Evaluation

For evalutaing the predicted article the customed class LtrPrediction is used from the prediction module.

```bash
from prediction import LtrPrediction

xgb_Prediction = LtrPrediction(model_bst, test_data)
results = xgb_Prediction.evaluate()
top_k_best_score_ips = xgb_Prediction.get_recomendation()

```

