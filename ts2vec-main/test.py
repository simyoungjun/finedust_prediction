import torch
import numpy as np
import argparse
import os
import sys
import time
import datetime
from ts2vec import TS2Vec
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout



out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)

pkl_save(f'{run_dir}/out.pkl', out)
pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
print('Evaluation result:', eval_res)

print("Finished.")