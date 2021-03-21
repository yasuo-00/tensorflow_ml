from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib
from make_input import make_input_fn
import model
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

def main():
    dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
    dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')

    CATEGORICAL_COLUMNS= ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
    NUMERIC_COLUMS = ['age', 'fare']

    feature_columns = []

    for feature_name in CATEGORICAL_COLUMNS:
        vocabulary = dftrain[feature_name].unique()
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

    for feature_name in NUMERIC_COLUMS:
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

    # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
    train_input_fn = make_input_fn(dftrain, y_train)
    eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

    linear_est = model.create_model(feature_columns)

    model.train_model(linear_est, train_input_fn, eval_input_fn)


if __name__ == "__main__":
    main()