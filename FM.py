
import tensorflow as tf
import argparse
from tensorflow import keras
from dataloader import dataloader

def parse_args():
    parser = argparse.ArgumentParser(description="NeuralMF.")
    parser.add_argument('--path', nargs='?', default='/dataset/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ratings.csv',
                        help='Choose a dataset.')
    parser.add_argument('--num_factors', type=int, default=8,help='latent feature of FM model.')
    parser.add_argument('--epochs', type=int, default=10,help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32,help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    return parser.parse_args()


class FM(keras.Model):
    def __init__(self, n_factor=8, **kwargs):
        super().__init__(**kwargs)

        self.w_0 = tf.Variable([0.0])
        self.w = tf.Variable(tf.zeros(shape=[p]))
        self.v = tf.Variable(tf.random.normal(shape=(p, n_factor)))

    def call(self,inputs):
        degree_1 = tf.reduce_sum(tf.multiply(self.w, inputs), axis=1)

        degree_2 = 0.5 * tf.reduce_sum(
            tf.math.pow(tf.matmul(inputs, self.v), 2)
            - tf.matmul(tf.math.pow(inputs, 2), tf.math.pow(self.v, 2))
            , 1
            , keepdims=False
        )

        predict = tf.math.sigmoid(self.w_0 + degree_1 + degree_2)

        return predict

def print_status_bar(iteration, total, loss, metrics = None):
    metrics = " - ".join([f"{m.name}: {m.result():.4f}"
                          for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"
    print(f"\r{iteration}/{total}  " + metrics ,
          end = end)

if __name__ == "__main__":

    args = parse_args()
    num_factors = args.num_factors
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size


    loader =dataloader(args.path + args.dataset)
    X_train =loader.X_train
    y_train = loader.y_train
    X_test = loader.X_test
    y_test = loader.y_test


    n = X_train.shape[0]
    p = X_train.shape[1]


    n_steps = len(X_train) // batch_size

    if learner.lower() == "adagrad":
        optimizer=keras.optimizers.Adagrad(lr=learning_rate)
    elif learner.lower() == "rmsprop":
        optimizer=keras.optimizers.RMSprop(lr=learning_rate)
    elif learner.lower() == "adam":
        optimizer=keras.optimizers.Adam(lr=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(lr=learning_rate)

    loss_fn = keras.losses.binary_crossentropy
    mean_loss = keras.metrics.Mean()
    metrics = [keras.metrics.BinaryAccuracy()]
    test_acc = keras.metrics.BinaryAccuracy()

    model = FM(n_factor=num_factors)

    train_data = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_train, tf.float32), tf.cast(y_train, tf.float32))).shuffle(500).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices(
        (tf.cast(X_test, tf.float32), tf.cast(y_test, tf.float32))).shuffle(200).batch(batch_size)

    for epoch in range(epochs):
        print(f"에포크 : {epoch}/{epochs}")

        for step, (X_batch, y_batch) in enumerate(train_data):
            # train, test data
            with tf.GradientTape() as tape:
                predict = model(X_batch)
                loss = loss_fn(y_batch, predict)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            mean_loss(loss)

            for metric in metrics:
                metric(y_batch, predict)

            print_status_bar(step * batch_size, len(y_train), mean_loss, metrics=metrics)

        for x_test, y_test in test_data:
            prediction = model(x_test)
            test_acc.update_state(y_test, prediction)

        print_status_bar(n_steps * batch_size, n_steps * batch_size, mean_loss, metrics=metrics)
        print("검증 정확도: ", test_acc.result().numpy())
        for metric in [mean_loss] + [test_acc] +metrics:
            metric.reset_states()

