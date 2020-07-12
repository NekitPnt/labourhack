#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from catboost import CatBoostClassifier, Pool


# In[2]:


from catboost import CatBoostRegressor
import catboost


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.linear_model as lm
import xgboost as xgb
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.model_selection import cross_val_score
import sklearn
from sklearn.metrics import roc_curve


# In[4]:


filename='vacs_train'

df=pd.read_csv(filename+'.csv', sep=';')


# In[7]:


filename='vacs_test'

df2=pd.read_csv(filename+'.csv', sep=';')


# In[35]:


df.columns


# In[19]:


from catboost.text_processing import Tokenizer
from catboost.text_processing import Dictionary


# In[16]:


def proc(series):

    tokenized = catboost.text_processing.Tokenizer(lowercasing=True,
                      separator_type='BySense',number_process_policy='Skip',
                      token_types=['Word', 'Number','Punctuation']).tokenize(series)
    return tokenized


# In[18]:


df['name.lemm.token']=df['name.lemm'].apply(proc)


# In[22]:


dictionary = Dictionary(occurence_lower_bound=0,
                        )\
    .fit(df['name.lemm.token'][0])

tokens = dictionary.get_top_tokens(14)
print (tokens)
dictionary.save("frequency_dict_path")


# In[28]:


dictionary.fit(df['name.lemm.token'][0])


# In[29]:


tokens = dictionary.get_top_tokens(14)
print (tokens)


# In[ ]:


def finder_1(string):
    if str(string).find('деревня')!=-1:
        return 1
    else: return 0
def finder_2(string):
    if str(string).find('посёлок')!=-1:
        return 1
    else: return 0
def finder_3(string):
    if str(string).find('село')!=-1:
        return 1
    else: return 0
def finder_4(string):
    if str(string).find('станица')!=-1:
        return 1
    else: return 0
def finder_5(string):
    if str(string).find('поселок')!=-1:
        return 1
    else: return 0


# In[ ]:





# In[31]:


from sklearn.feature_extraction.text import CountVectorizer


# In[34]:


vectorizer = CountVectorizer()
data_corpus = df['name.lemm.token'][0]
X = vectorizer.fit_transform(data_corpus) 
print(X.toarray())
print(vectorizer.get_feature_names())


# In[ ]:


y = df['salary_from']
X = df[featurecolumns2[10:]]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1234)


# In[ ]:


nkill_dop=[]
for i in df3.index:
    nkill_dop.append('Nan')


# In[ ]:


X


# In[ ]:


model = CatBoostRegressor(iterations=5000,
                           #depth=4,
                           #learning_rate=0.17,
                           loss_function='RMSE',
                           eval_metric='R2',
                           verbose=True,
                           use_best_model=True,
                           #custom_loss=['R2', 'RMSE'],
                           #l2_leaf_reg=1.7,
                           nan_mode='Min',
                           #sampling_frequency='PerTreeLevel',
                           #leaf_estimation_method='Simple',
                           #boosting_type='Ordered'
                          )


# In[ ]:


model = CatBoostRegressor(iterations=2000,
                           depth=5,
                           learning_rate=0.03,
                           loss_function='RMSE',
                           eval_metric='R2',
                           verbose=True,
                           use_best_model=True,
                           #custom_loss=['AUC', 'Accuracy'],
                           l2_leaf_reg=2.7,
                           nan_mode='Max',
                           #sampling_frequency='PerTreeLevel',
                           #leaf_estimation_method='Simple',
                           #boosting_type='Ordered'
                          )


# In[ ]:


model.fit(X_train, y_train, plot=True, silent=True, eval_set=(X_test, y_test), use_best_model=True, early_stopping_rounds=100)


# In[ ]:


from catboost import FeaturesData


# In[ ]:


import catboost


# In[ ]:


model.get_feature_importance(prettified=True)


# In[ ]:


forplot=model.get_feature_importance(prettified=True)


# In[ ]:



for i in range(len(forplot)):
    if forplot['Importances'][i]==0:
        thrash2.append(forplot['Feature Id'][i])
        


# In[ ]:


thrash2


# In[ ]:


impsum=[]
scht=0
for i in forplot['Importances']:
    scht+=i
    impsum.append(scht)
forplot['impsum']=impsum


# In[ ]:


forplot.to_csv('for_indexingv3.csv', sep=';')


# In[ ]:


forplot['impsum'].plot()


# In[ ]:


forplot.plot()


# In[ ]:


predictions=model.predict(df2018[featurecolumns2[10:]])


# In[ ]:


df2018['instability_index'].describe().to_csv('descr.csv', sep=';')


# In[ ]:


model.predict(df2018[featurecolumns2[9:]])


# In[ ]:


index_result=[]
for i in range(len(predictions)):
    index_result.append(predictions[i]*1000000)


# In[ ]:


df2018['instability_index']=index_result


# In[ ]:


df2018['instability_index']=df2018['instability_index']-df2018['instability_index'].min()


# In[ ]:


def ind_rec(series):
    if series<=35: return 'Low'
    if 35<series<=65: return 'Middle'
    if series>65: return 'High'


# In[ ]:


df2018['instability_index_rec']=df2018['instability_index'].apply(ind_rec)


# In[ ]:


df2018.to_csv('with_ind_alll_fac1-3_pop.csv', index=False, sep=';')


# In[ ]:


model.get_feature_importance(prettified=True).to_csv('FAC1_3'+'1992_important.csv', sep=';')


# In[ ]:


pool=Pool(data=X, label=y)


# In[ ]:


shap_values = model.get_feature_importance(pool, type='ShapValues')

expected_value = shap_values[0,-1]
shap_values = shap_values[:,:-1]

print(shap_values.shape)


# In[ ]:


import shap

shap.initjs()
shap.force_plot(expected_value, shap_values[3,:], X.iloc[3,:])


# In[ ]:


import shap

shap.initjs()
shap.force_plot(expected_value, shap_values[91,:], X.iloc[91,:])


# In[ ]:


shap.summary_plot(shap_values, X)


# In[ ]:


popo=['year','c_names','v2clrelig']


# In[ ]:


pop2=df[popo]


# In[ ]:


pop2.to_csv('yyyyy.csv',sep=';')


# In[ ]:


df['v2clrelig'].describe()


# In[ ]:


res = model.calc_feature_statistics(X,
                                    y,
                                    feature='us_foreign_aid',
                                    plot=True)


# In[ ]:


df2na.describe()

