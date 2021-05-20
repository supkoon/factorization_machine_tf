import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from dataloader import dataloader




class FM(keras.Model):

    def __init__(self,num_factor,num_features,**kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.num_factor = num_factor
        self.w_0 = tf.Variable([0.0])
        self.w = tf.Variable(tf.zeros([self.num_features]))
        self.v = tf.Variable(tf.random.normal(shape = (self.num_features,self.num_factor)))


    def call(self,inputs):
        degree_1 = tf.reduce_sum(tf.math.multiply(self.w, inputs),axis=1)
        degree_2 = 0.5 * tf.reduce_sum(

            tf.math.pow(tf.matmul(inputs,self.v),2)
            -tf.matmul(tf.math.pow(inputs,2),tf.math.pow(self.v,2)),
        axis=1,
        keepdims=False
        )
        y_hat = tf.math.sigmoid(self.w_0 + degree_1 + degree_2)

        return y_hat

def train_on_batch(model, optimizer, accuracy, inputs, targets):
    with tf.GradientTape() as tape:
        y_pred = model(inputs)
        loss = tf.keras.losses.MSE(                y_true=targets,
                                                   y_pred=y_pred)

    # loss를 모델의 파라미터로 편미분하여 gradients를 구한다.
    grads = tape.gradient(target=loss, sources=model.trainable_variables)

    # apply_gradients()를 통해 processed gradients를 적용한다.
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # accuracy: update할 때마다 정확도는 누적되어 계산된다.
    accuracy.update_state(targets, y_pred)

    return loss

# 반복 학습 함수
def train(epochs):
    loader=dataloader("datasets/movielens")
    X_train,Y_train = loader.generate_trainset()
    X_test,Y_test = loader.generate_testset()
    train_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train, tf.float32), tf.cast(Y_train, tf.float32))).shuffle(500).batch(8)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_test, tf.float32), tf.cast(Y_test, tf.float32))).shuffle(200).batch(8)
    num_factors = 8
    num_features = X_train.shape[1]

    model = FM(num_factors,num_features)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    accuracy = keras.metrics.BinaryAccuracy(threshold=0.5)
    loss_history = []

    for i in range(epochs):
        for x, y in train_ds:
            loss = train_on_batch(model, optimizer, accuracy, x, y)
            loss_history.append(loss)

        if i % 2 == 0:
            print("스텝 {:03d}에서 누적 평균 손실: {:.4f}".format(i, np.mean(loss_history)))
            print("스텝 {:03d}에서 누적 정확도: {:.4f}".format(i, accuracy.result().numpy()))

    test_accuracy = keras.metrics.BinaryAccuracy(threshold=0.5)
    for x, y in test_ds:
        y_pred = model(x)
        test_accuracy.update_state(y, y_pred)

    print("테스트 정확도: {:.4f}".format(test_accuracy.result().numpy()))

train(10)