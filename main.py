import argparse
import datetime
from tensorflow import keras

from model import NARM
from utils.myCallback import HistoryRecord, P_MRR

parser = argparse.ArgumentParser()
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate')
parser.add_argument('--lr_dc_step', type=float, default=0.5, help='learning rate')
parser.add_argument('--layer', type=float, default=1, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.001, help='ssl task maginitude')
opt = parser.parse_args()
print(opt)





model = NARM(emb_size=opt.embSize)
save_dir = 'logs'
time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=opt.lr,
                                                          decay_rate=opt.lr_dc,
                                                          decay_steps=opt.lr_dc_step * epoch_steps,
                                                          staircase=False)
early_stopping = keras.callbacks.EarlyStopping(monitor='MRR@20',
                                               min_delta=0,
                                               patience=5,
                                               verbose=1,
                                               mode='max')
history_recoder = HistoryRecord(log_dir=os.path.join(save_dir, 'log_' + time_str))
p_mrr = P_MRR(val_data=test_dataloader, performance_mode=2, val_size=int(test_data_size/100))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              run_eagerly=False,
              jit_compile=False)
model.fit(x=,
          epochs=30,
          verbose=1,
          callbacks=[p_mrr, early_stopping, history_recoder],
          validation_data=)
