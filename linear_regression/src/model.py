import tensorflow as tf

from IPython.display import clear_output

def create_model(feature_columns):
    #age_x_gender = tf.feature_column.crossed_column(['age', 'sex'], hash_bucket_size=100)
    #derived_feature_columns = [age_x_gender]
    #linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns+derived_feature_columns)
    linear_est = tf.estimator.LinearClassifier(feature_columns = feature_columns)

    return linear_est


def train_model(linear_est, train_input_fn, eval_input_fn):
    linear_est.train(train_input_fn)
    result = linear_est.evaluate(eval_input_fn)

    clear_output()
    print(result['accuracy'])