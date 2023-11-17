from functools import partial
from itertools import chain
from tqdm import tqdm

import numpy as np
import tensorflow as tf


def generate_data(datas):
    x = datas[0]
    y = datas[1]
    for sample, label in zip(x, y):
        yield sample, label  # 生成几个，流式处理接口就放几个，这里并没有按照 输出定义 输出tuple，而是单个的


def process_data(x, y):
    session_len = tf.cast(tf.shape(x)[0], tf.float32)
    items = x
    mask = tf.ones(shape=(session_len,))

    sample = (items, mask)
    labels = y - 1

    return sample, labels


def compute_max_len(raw_data):
    x = raw_data[0]
    # 找出最长的序列长度
    len_list = [len(d) for d in x]
    max_len = np.max(len_list)
    return max_len


# def compute_item_num(sequence):
#     x = sequence[0]
#     y = sequence[1]
#     x_seq_in_1D = list(chain.from_iterable(x))
#     items_num = len(np.unique(x_seq_in_1D))
#     items_num += len(np.unique(y))
#     return items_num


def compute_max_node(sequence):
    seq_in_1D = list(chain.from_iterable(sequence))
    max_node = np.max(seq_in_1D)
    return max_node


def split_train_val(train_data, split_rate=0.1):
    session_total = len(train_data[0])
    split_num = int(session_total * split_rate)
    val_index = np.random.choice(a=np.arange(0, session_total), size=split_num, replace=False)
    np.random.shuffle(val_index)
    val_data = ([train_data[0][index] for index in val_index], [train_data[1][index] for index in val_index])
    # train_index = np.setdiff1d(np.arange(0, session_total), val_index)
    # train_data_new = ([], [])
    # for index in tqdm(train_index, total=len(train_index), desc='分割中'):
    #     train_data_new[0].append(train_data[0][index])
    #     train_data_new[1].append(train_data[1][index])
    return val_data


class DataLoader:
    def __init__(self, raw_data, train_mode=True):
        self.max_len = compute_max_len(raw_data)  # 最长序列
        self.data = raw_data
        # self.data = self.reverse_data()  # 反转输入序列
        self.train_mode = train_mode

    def dataloader(self):
        dataset = tf.data.Dataset.from_generator(generator=partial(generate_data, self.data),
                                                 output_signature=(tf.TensorSpec(shape=None,
                                                                                 dtype=tf.int32),
                                                                   tf.TensorSpec(shape=(),
                                                                                 dtype=tf.int32)))  # (x, label)
        dataset = dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        if self.train_mode:
            # pass
            # # TODO： 训练时打开shuffle，调试时避免减损性能
            dataset = dataset.shuffle(buffer_size=len(self.data[0]) - (len(self.data[0]) % 100))
        dataset = dataset.padded_batch(batch_size=100,  # (session_len, items, reversed_items, mask)
                                       padded_shapes=(
                                           ([self.max_len],
                                            [self.max_len],
                                            ),
                                           []),
                                       drop_remainder=True
                                       )
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


if __name__ == '__main__':
    process_data([1, 2, 3, 4], 5)
