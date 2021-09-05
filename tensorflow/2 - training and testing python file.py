#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


# read the training dataset and save to a dataframe
x_train = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv");

#below reads the validation dataset
x_eval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv");

# you shoud have train and validation/test to see(test) if your algorithm is doing well. 
# you will train with x_train and validate with x_eval because it has not seen x_eval and there will beless bias.


# we want to know if a passenger will survive... 
# ...given specific features or characteristics (X) about a passenger
# for that we will use SURVIVED as our output(y) since this is what we are trying to predict.
# so lets put survived in y and remove it from our dataframe using pop

y_train = x_train.pop("survived")
y_eval = x_eval.pop("survived")


# In[4]:


# categorical values have a limited/fixed number of possible values
# numeric columns take on numeric values with different possible values 
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = x_train[feature_name].unique() #find uniquevalues in that column e.g male, female
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))


# In[5]:


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_fn(x_train, y_train)
eval_input_fn = make_input_fn(x_eval, y_eval, num_epochs=1, shuffle=False)


# In[ ]:


#using a Linear classifier
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

#training
linear_est.train(train_input_fn)

#evaluate and store in result
result = linear_est.evaluate(eval_input_fn)

clear_output()
#print result
print(result)

#print first result
print(result[0])

#print first result probability
print(result[0]["probabilities"])

#print first result cances of survival
print(result[0]["probabilities"][1])

#print first persons features
print(x_eval.loc[0])

#print first persons actual probability of surviving
print(y_eval.loc[0])

