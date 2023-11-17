import argparse
import datetime
import os
import pickle
from tensorflow import keras

from model import NARM
from utils.myCallback import HistoryRecord, P_MRR
from utils.dataloader import split_train_val, DataLoader, compute_max_len

parser = argparse.ArgumentParser()
parser.add_argument('--embSize', type=int, default=50, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate')
parser.add_argument('--lr_dc_step', type=float, default=3, help='learning rate')
opt = parser.parse_args()
print(opt)

train_data = pickle.load(open('dataset/diginetica/train.txt', 'rb'))
test_data = pickle.load(open('dataset/diginetica/test.txt', 'rb'))
val_data = split_train_val(train_data, split_rate=0.1)
item_num = 43097
print("item总数量{}".format(item_num))
train_data_size = len(train_data[0])
test_data_size = len(test_data[0])
val_data_size = len(val_data[0])
epoch_steps = int(train_data_size/100)
max_len = compute_max_len(train_data)

train_dataloader = DataLoader(train_data).dataloader()
test_dataloader = DataLoader(test_data, train_mode=False).dataloader()
val_dataloader = DataLoader(val_data, train_mode=False).dataloader()

# MODEL
model = NARM(emb_size=opt.embSize, n_node=item_num, max_len=max_len)
save_dir = 'logs'
time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=opt.lr,
                                                          decay_rate=opt.lr_dc,
                                                          decay_steps=opt.lr_dc_step * epoch_steps,
                                                          staircase=True)
early_stopping = keras.callbacks.EarlyStopping(monitor='MRR@20',
                                               min_delta=0,
                                               patience=5,
                                               verbose=1,
                                               mode='max')
history_recoder = HistoryRecord(log_dir=os.path.join(save_dir, 'log_' + time_str))
p_mrr = P_MRR(val_data=test_dataloader, performance_mode=1, val_size=int(test_data_size / 100))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              run_eagerly=False,
              jit_compile=False)
model.fit(x=train_dataloader,
          epochs=30,
          verbose=1,
          callbacks=[p_mrr, early_stopping, history_recoder],
          validation_data=val_dataloader)
