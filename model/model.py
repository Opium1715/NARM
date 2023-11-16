import tensorflow as tf
from tensorflow import keras


class GlobalEncoder(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.emb_size = kwargs.get('emb_size')
        self.gru = keras.layers.GRU(units=self.emb_size,
                                    use_bias=True,
                                    return_state=True)

    def call(self, inputs, *args, **kwargs):
        return self.gru(inputs)


class LocalEncoder(keras.layers.Layer):
    def __init__(self, emb_size, **kwargs):
        super().__init__(**kwargs)
        self.emb_size = emb_size
        self.A1 = keras.layers.Dense(units=self.emb_size,
                                     use_bias=False)
        self.A2 = keras.layers.Dense(units=self.emb_size,
                                     use_bias=False)
        self.v = self.add_weight(shape=(1, self.emb_size),
                                 dtype=tf.float32,
                                 name='v',
                                 initializer='uniform')
        self.gru = keras.layers.GRU(units=self.emb_size,
                                    use_bias=True,
                                    return_state=True,
                                    return_sequences=True)

    def call(self, inputs, *args, **kwargs):
        # soft_attention
        # hidden_t = kwargs.get('hidden_t')
        session_hidden = kwargs.get('session_hidden')
        batch_size = tf.shape(session_hidden)[0]
        last_state, all_state = self.gru(session_hidden)
        # compute the coef of attention
        multi_last_state = tf.tile(last_state, multiples=(1, batch_size, 1))
        alpha = tf.matmul(self.v, tf.sigmoid(self.A1(multi_last_state) + self.A2(all_state)))
        return tf.reduce_sum(alpha * all_state, axis=1)


class NARM(keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = kwargs.get('emb_size')
        self.n_node = kwargs.get('n_node')
        self.item_embeddings = keras.layers.Embedding(self.n_node, self.emb_size)
        self.globalEncoder = GlobalEncoder(emb_size=self.emb_size)
        self.localEncoder = LocalEncoder(emb_size=self.emb_size)
        self.B = self.add_weight(shape=(self.emb_size, self.emb_size),
                                 name='B',
                                 dtype=tf.float32)

    def call(self, inputs, training=None, mask=None):
        session_hidden = inputs
        Ct_global = self.globalEncoder(session_hidden)
        Ct_local = self.localEncoder(session_hidden)
        Ct = tf.concat([Ct_global, Ct_local], axis=0)

        similarity_score = tf.matmul(tf.matmul(self.item_embeddings(tf.range(self.n_node)),
                                     self.B),
                                     Ct)

        return similarity_score


