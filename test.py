import argparse, os

import torch

from src.util.config_parse import ConfigParser
from src.trainer import get_trainer_class


def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('-s', '--session_name', default= 'APBSN_SIDD' ,  type=str)  #'APBSN_SIDD'  'APBSN_DND'
    args.add_argument('-c', '--config',       default="/home/lhy008/AP-BSN/conf/APBSN_SIDD" ,  type=str)  #"/home/lhy008/AP-BSN/conf/APBSN_SIDD"  "/home/lhy008/AP-BSN/conf/APBSN_DND"
    args.add_argument('-e', '--ckpt_epoch',   default=28,     type=int)
    args.add_argument('-g', '--gpu',          default='3',  type=str)  #None
    args.add_argument('--pretrained',   default=None,  type=str)
    args.add_argument('--thread',       default=4,     type=int)
    args.add_argument('--self_en',      action='store_true')
    args.add_argument('--test_img',     default=None,  type=str)
    args.add_argument('--test_dir',     default=None,  type=str)

    args = args.parse_args()

    assert args.config is not None, 'config file path is needed'
    if args.session_name is None:
        args.session_name = args.config # set session name to config file name

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    # intialize trainer
    trainer = get_trainer_class(cfg['trainer'])(cfg)

    # test
    trainer.test()


if __name__ == '__main__':
    main()
