#!/usr/bin/python
#coding:utf-8

import utils
import argparse
import os
import stat
import json
import traceback

parser = argparse.ArgumentParser()

parser.add_argument('--config', help='config file',
                    default='config.json')
parser.add_argument('--gpu', help='which gpu to use', default='2')
parser.add_argument("--is_train", type=utils.str2bool, default=True, help="is_train or infer&evaluate")
parser.add_argument("--shuffle_data", type=utils.str2bool, default=False, help="is_train or infer&evaluate")
parser.add_argument('--seeds', help='rand seed to use', default='10,20,30')

args = parser.parse_args()
logger = utils.get_logger(os.path.join(os.path.dirname(args.config), 'multi_seed.log'))

config = utils.load_config(args.config)
seeds = [i.strip() for i in args.seeds.split(',') if len(i.strip()) != 0]

base_dir = os.path.dirname(args.config)
data = os.path.join(base_dir, 'data')
preprocess = os.path.join(base_dir, 'preprocess')

if not os.path.isdir(data):
    logger.error('cannot find data file')
    exit(1)


def chmod444(p):
    if os.path.isfile(p):
        os.chmod(p, stat.S_IROTH | stat.S_IRGRP | stat.S_IRUSR)
    else:
        for file in os.listdir(p):
            chmod444(os.path.join(p, file))


def chmod664(p):
    if os.path.isfile(p):
        os.chmod(p, stat.S_IROTH | stat.S_IRGRP | stat.S_IWGRP | stat.S_IRUSR | stat.S_IWUSR)
    else:
        for file in os.listdir(p):
            chmod664(os.path.join(p, file))


def run_cmd(cmd, logger):
    logger.info("cmd: {}".format(cmd))
    os.system(cmd)


try:
    if args.is_train:
        for seed in seeds:
            logger.info('handling seed {}'.format(seed))
            model_dir = os.path.join(base_dir, 'seed_{}'.format(seed))
            config_file = os.path.join(model_dir, 'config.json')
            if not os.path.isdir(model_dir):
                logger.info('mkdir {}'.format(model_dir))
                os.makedirs(model_dir)
            else:
                logger.info('{} exists'.format(model_dir))
            config['poj_base'] = model_dir
            logger.info('copying data to {}'.format(model_dir))
            run_cmd('cp {} {} -r'.format(data, model_dir), logger)
            if not args.shuffle_data and os.path.isdir(preprocess):
                logger.info('copying preprocessed data to {}'.format(model_dir))
                os.system('cp {} {} -r'.format(preprocess, model_dir))

            config['seed'] = int(seed)
            with open(config_file, 'w') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            logger.info('training with seed {}'.format(seed))

            train_cmd = 'python3 main.py --is_train {train} --config {config} --gpu {gpu}'.format(
                train=str(args.is_train), config=config_file, gpu=args.gpu)

            run_cmd(train_cmd, logger)
            if not args.shuffle_data and not os.path.isdir(preprocess):
                tar_dir = os.path.join(model_dir, 'preprocess')
                logger.info('copying {} to {}'.format(tar_dir, model_dir))
                run_cmd('cp {} {} -r'.format(tar_dir, base_dir), logger)
            logger.info("change files in {} to 444".format(model_dir))
            chmod444(model_dir)

    else:
        for seed in seeds:
            logger.info('handling seed {}'.format(seed))
            model_dir = os.path.join(base_dir, 'seed_{}'.format(seed))
            log_file = [i for i in os.listdir(model_dir) if i.endswith('.log')]
            for file in log_file:
                chmod664(os.path.join(model_dir, file))
            chmod664(os.path.join(model_dir, 'eval'))
            chmod664(os.path.join(model_dir, 'preprocess'))
            config_file = os.path.join(model_dir, 'config.json')
            logger.info('test {}'.format(config_file))
            test_cmd = 'python3 main.py --is_train {train} --config {config} --gpu {gpu}'.format(
                train=str(args.is_train), config=config_file, gpu=args.gpu)
            run_cmd(test_cmd, logger)

except:
    logger.error(traceback.format_exc())






