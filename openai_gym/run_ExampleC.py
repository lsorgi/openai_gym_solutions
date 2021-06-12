from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K

import tensorflow as tf
import numpy as np


class ExampleC(object):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims,
                 learning_rate,
                 memory_sz,
                 batch_sz):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.memory_sz = memory_sz
        self.batch_size = batch_sz
        self.input_nd_memory = np.zeros(shape=(self.memory_sz, self.input_dim), dtype=float)
        self.output_nd_memory = np.zeros(shape=(self.memory_sz, self.output_dim), dtype=float)
        self.output_1d_memory = np.zeros(shape=self.memory_sz, dtype=float)
        self.memory_index = 0
        self.step_index = 0
        # build model1 (input_nd -> output_nd)
        input_nd = layers.Input(shape=(self.input_dim,))
        hidden = layers.Dense(units=hidden_dims[0], activation='relu')(input_nd)
        for jh in np.arange(1, len(hidden_dims)):
            hidden = layers.Dense(units=hidden_dims[jh], activation='relu')(hidden)
        output_nd = layers.Dense(units=self.output_dim, activation='tanh')(hidden)
        self.model1 = models.Model(input=[input_nd], output=[output_nd])
        self.model1.summary()
        # build model2 ( [input_nd, output_nd] -> output_1d)
        output_nd_ = layers.Input(shape=(self.output_dim,))
        conc = layers.concatenate([input_nd, output_nd_])
        hidden = layers.Dense(units=hidden_dims[0], activation='relu')(conc)
        for jh in np.arange(1, len(hidden_dims)):
            hidden = layers.Dense(units=hidden_dims[jh], activation='relu')(hidden)
        output_1d = layers.Dense(units=1, activation='sigmoid')(hidden)
        self.model2 = models.Model(input=[input_nd, output_nd_], output=[output_1d])
        self.model2.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
        self.model2.summary()

        def train_function(input_nd: np.array):
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            input_nd_var = tf.convert_to_tensor(input_nd, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(self.model1.trainable_weights)
                pred_output_nd = self.model1(input_nd_var)
                pred_output_1d = self.model2([input_nd_var, pred_output_nd])
                learn_loss = -K.log(K.clip(pred_output_1d, 1e-8, 1.0))
                grads = tape.gradient(learn_loss, self.model1.trainable_weights)
                opt.apply_gradients(zip(grads, self.model1.trainable_weights))

        self.train_function = train_function

    def get(self, input_nd: np.array):
        return self.model1.predict(x=input_nd[np.newaxis, :])[0]

    def push(self, input_nd: np.array, output_nd: np.array, output_1d: np.float):

        self.input_nd_memory[self.memory_index] = input_nd
        self.output_nd_memory[self.memory_index] = output_nd
        self.output_1d_memory[self.memory_index] = output_1d
        # train model1
        self.train_function(input_nd[np.newaxis, :])
        # train model2
        if self.step_index >= self.batch_size:
            js = np.random.uniform(
                low=0,
                high=self.step_index,
                size=self.batch_size).astype(int) % self.memory_sz
            input_nd_batch = self.input_nd_memory[js, :]
            output_nd_batch = self.output_nd_memory[js, :]
            output_1d_batch = self.output_1d_memory[js]
            cost = self.model2.train_on_batch(x=[input_nd_batch, output_nd_batch], y=output_1d_batch)
        #
        self.memory_index = (self.memory_index + 1) % self.memory_sz
        self.step_index += 1


def run():
    controller = ExampleC(
        input_dim=3,
        output_dim=1,
        hidden_dims=[8, 8],
        learning_rate=1e-3,
        memory_sz=5000,
        batch_sz=32)

    while True:
        input_nd = np.random.uniform(0, 1, size=3)
        output_nd = controller.get(input_nd)
        output_1d = np.random.uniform(0, 1)
        controller.push(input_nd, output_nd, output_1d)


if __name__ == "__main__":
    run()