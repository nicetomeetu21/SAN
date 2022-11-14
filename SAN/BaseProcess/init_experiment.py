# -*- coding:utf-8 -*-
import os
import pytorch_lightning as pl
from natsort import natsorted
import shutil
import sys

def gen_train_data_list(total_data_path,test_data_list):
    total_list = natsorted(os.listdir(total_data_path))
    total_list = [name[:-4] for name in total_list]
    for name in test_data_list:
        total_list.remove(name)
    return total_list

def initExperiment(opts):
    opts.default_root_dir = os.path.join(opts.result_root, opts.exp_name, str(opts.fold_id))
    if opts.fold_id < 3:
        opts.test_data_list = opts.test_data_lists[opts.fold_id]
        opts.train_data_list = gen_train_data_list(opts.image_root, opts.test_data_list)
    else:
        opts.test_data_list = opts.test_data_lists[opts.fold_id]
        opts.train_data_list = opts.test_data_list

    if opts.reproduce:
        pl.seed_everything(42, workers=True)
        opts.deterministic = opts.reproduce
        opts.benchmark = not opts.reproduce

    if opts.command == 'fit':
        if not os.path.exists(opts.default_root_dir):
            os.makedirs(opts.default_root_dir)
            code_dir = os.path.abspath(os.path.dirname(os.getcwd()))
            shutil.copytree(code_dir, os.path.join(opts.default_root_dir, 'code'))
        else:
            sys.exit("result_dir exists: "+opts.default_root_dir)