import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Model

# 加上下面這幾行
import mlflow
from mlflow import pyfunc
import mlflow.tensorflow
from mlflow import log_metric, log_param, log_artifacts
import os


print(tf.__version__)


# Load in the data
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 縮小一點for demo 用
x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]

# Increase one dimension so it can be used by the 2D convolutional keras layer
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train.shape:", x_train.shape)


# ---------------------------
# Run the Model
# ---------------------------

# 加這行
mlflow.tensorflow.autolog()


def run_model(experiment_id, params):

    run_id = params["run_id"]
    run_name = params["run_name"]

    with mlflow.start_run(experiment_id=experiment_id,
                          # run_id=run_id,
                          run_name=run_name) as run:

        # log parameter
        log_param("epochs", params["epochs"])
        log_param("convSize", params["convSize"])

        # number of classes
        K = len(set(y_train))
        print("number of classes:", K)
        # Build the model using the functional API
        i = Input(shape=x_train[0].shape)
        x = Conv2D(32, params['convSize'], strides=2, activation='relu')(i)
        x = Conv2D(64, params['convSize'], strides=2, activation='relu')(x)
        x = Conv2D(128, params['convSize'], strides=2, activation='relu')(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(K, activation='softmax')(x)

        model = Model(i, x)

        # Compile and fit
        # Note: make sure you are using the GPU for this!
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        r = model.fit(x_train, y_train, validation_data=(
            x_test, y_test), epochs=params['epochs'])

        # log metric
        log_metric("foo", 12345)
        log_metric("bar", 6789)

        # log artifact
        run_folder = f"outputs/{params['run_name']}"
        if not os.path.exists(run_folder):
            os.makedirs(run_folder)
        aa = pd.DataFrame({
            "foo": [123, 456],
            "bar": ["hahaha", "hihihi"],
            "epochs": params["epochs"]
        })
        out_path = os.path.join(run_folder, "output.csv")
        aa.to_csv(out_path)
        log_artifacts(run_folder)

        return (run.info.experiment_id, run.info.run_id)


# 實驗開始
experiment_id = "select_model_structure"
params_list = [{'run_name': "run_name_1", 'run_id': 'run_id_1', 'epochs': 1, 'convSize': 2},
               {'run_name': "run_name_2", 'run_id': 'run_id_2', 'epochs': 10, 'convSize': 3}]

for params in params_list:
    run_model(experiment_id, params)
    print("Done!!")


# for epochs, convSize in [[3,2], [15,3]]:
#   params = {'epochs': epochs,
#             'convSize': convSize}
#   run_model(params)
