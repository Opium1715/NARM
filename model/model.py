import tensorflow as tf
from tensorflow import keras


class GlobalEncoder(keras.layers.Layer):
    def __init__(self, emb_size, **kwargs):
        super().__init__(**kwargs)
        self.emb_size = emb_size
        self.gru = keras.layers.GRU(units=100,
                                    use_bias=True)

    def call(self, inputs, *args, **kwargs):
        mask = tf.cast(inputs[1], tf.bool)
        return self.gru(inputs=inputs[0], mask=mask)


class LocalEncoder(keras.layers.Layer):
    def __init__(self, emb_size, max_len, **kwargs):
        super().__init__(**kwargs)
        self.emb_size = emb_size
        self.max_len = max_len
        self.A1 = keras.layers.Dense(units=self.emb_size,
                                     use_bias=False)
        self.A2 = keras.layers.Dense(units=self.emb_size,
                                     use_bias=False)
        self.v = self.add_weight(shape=(1, self.emb_size),
                                 dtype=tf.float32,
                                 name='v',
                                 initializer='uniform')
        self.gru = keras.layers.GRU(units=100,
                                    use_bias=True,
                                    return_state=True,
                                    return_sequences=True)

    def call(self, inputs, *args, **kwargs):
        # soft_attention
        session_hidden = inputs[0]
        mask = inputs[1]
        bool_mask = tf.cast(mask, tf.bool)
        time_step = tf.shape(session_hidden)[1]
        all_state, last_state = self.gru(inputs=session_hidden, mask=bool_mask)
        # compute the coef of attention
        multi_last_state = tf.reshape(tf.tile(last_state, multiples=(1, time_step)),
                                      shape=(-1, time_step, self.emb_size))
        alpha = tf.squeeze(
            tf.matmul(self.v, tf.sigmoid(self.A1(multi_last_state * tf.expand_dims(mask, -1)) + self.A2(all_state)),
                      transpose_b=True),
            axis=1)
        return tf.reduce_sum(tf.expand_dims(alpha, -1) * all_state, axis=1)


class NARM(keras.models.Model):
    def __init__(self, emb_size, n_node, max_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_size = emb_size
        self.n_node = n_node
        self.item_embeddings = keras.layers.Embedding(self.n_node + 1, self.emb_size,
                                                      mask_zero=True,
                                                      embeddings_initializer=tf.keras.initializers.HeUniform())
        self.globalEncoder = GlobalEncoder(emb_size=self.emb_size)
        self.localEncoder = LocalEncoder(emb_size=self.emb_size, max_len=max_len)
        self.B = self.add_weight(shape=(self.emb_size, 200), # 2*batch_size
                                 initializer='uniform',
                                 name='B',
                                 dtype=tf.float32)
        self.dropout_1 = keras.layers.Dropout(rate=0.25)
        self.dropout_2 = keras.layers.Dropout(rate=0.5)

    def call(self, inputs, training=None, mask=None):
        session = inputs[0]
        mask = inputs[1]
        session_hidden = self.item_embeddings(session)
        # session_hidden = tf.gather(self.embedding_test, session)
        drop_session_hidden = self.dropout_1(session_hidden)
        Ct_global = self.globalEncoder((drop_session_hidden, mask))
        Ct_local = self.localEncoder((drop_session_hidden, mask))
        Ct = tf.concat([Ct_global, Ct_local], axis=0)
        drop_Ct = self.dropout_2(Ct)
        similarity_score = tf.matmul(tf.matmul(self.B, drop_Ct),
                                     # tf.gather(self.embedding_test, tf.range(1, self.n_node + 1)),
                                     self.item_embeddings(tf.range(1, self.n_node + 1)),
                                     transpose_b=True)
        # output = tf.nn.softmax(similarity_score)
        return similarity_score
