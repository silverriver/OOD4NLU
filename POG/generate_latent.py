import tensorflow as tf
import utils
import argparse
from model import CGAN
import os
import time
import traceback
import gen_tfrecord
import numpy as np

logger = utils.get_logger('main.log')
parser = argparse.ArgumentParser()

parser.add_argument('--config', help='config file',
                    default='config.json')
parser.add_argument('--gpu', help='which gpu to use', default='0')
parser.add_argument('--ind_infile', help='path to the input file of ind',
                    default='data/ind_dev')
parser.add_argument('--ood_infile', help='path to the input file of ind',
                    default='data/ood_dev')
parser.add_argument('--ind_outfile', help='path to the output file of ind', default='/tmp/ind_outfile')
parser.add_argument('--ood_outfile', help='path to the output file of ood', default='/tmp/ood_outfile')
parser.add_argument('--gen_outfile', help='path to the output file of gen', default='/tmp/gen_outfile')
parser.add_argument('--gen_count', help='how many data to generate', type=int, default=1000)
parser.add_argument('--gen_batch_size', help='batch size for ood', type=int, default=500)
parser.add_argument('--ind_batch_size', help='batch size for ind', type=int, default=100)

args = parser.parse_args()
config = utils.load_config(args.config)

data_dir = os.path.join(config['poj_base'], config['data_dir'])
train_dir = os.path.join(config['poj_base'], config['train_dir'])
preprocess_dir = os.path.join(config['poj_base'], config['preprocess_dir'])

word_vocab_file = os.path.join(preprocess_dir, config['word_vocab_file'])
intent_vocab_file = os.path.join(preprocess_dir, config['intent_vocab_file'])
ind_outfile = args.ind_outfile
ind_infile = args.ind_infile
ood_infile = args.ood_infile
ood_outfile = args.ood_outfile
gen_outfile = args.gen_outfile
gen_count = args.gen_count
gen_batch_size = args.gen_batch_size
ind_batch_size = args.ind_batch_size

np.random.seed(config['seed'])


try:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info('tf Version: {}'.format(tf.__version__))
    for i in config:
        logger.info('{}: {}'.format(i, config[i]))
    for i in vars(args):
        logger.info('{}: {}'.format(i, getattr(args, i)))

    ckpt_file = tf.train.latest_checkpoint(train_dir)
    if not ckpt_file:
        logger.error("cannot find checkpoint in {}".format(train_dir))
        exit(1)

    logger.info("found checkpoint {}".format(ckpt_file))

    word2id, id2word = utils.read_vocab(word_vocab_file, logger, config['word_vocab_size'])
    intent2id, id2intent = utils.read_vocab(intent_vocab_file, logger)
    config = utils.update_vocab_size(config, intent2id, dict())

    ind_valid_data = gen_tfrecord.load_data(
        word2id, intent2id, [ind_infile], logger, config['max_utter_len'], word_level=True, shuffle=True)

    ood_valid_data = gen_tfrecord.load_data(
        word2id, intent2id, [ood_infile], logger, config['max_utter_len'], word_level=True, shuffle=True)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Session(config=tf_config) as sess:
        logger.info('loading model from {}'.format(ckpt_file))
        model = CGAN(config=config, go_id=utils.GO_ID, eos_id=utils.EOS_ID)
        model.saver.restore(sess, ckpt_file)

        class print_logger():
            def info(self, str):
                print(str)

        p_logger = print_logger()
        ind_count = len(ind_valid_data['intent'])
        logger.info('generating {} ind samples to {}'.format(ind_count, ind_outfile))
        with open(ind_outfile, 'w') as f:
            curr = 0
            while curr < ind_count:
                enc_h = sess.run(model.enc_h,
                                 feed_dict={model.utter: ind_valid_data['utter'][curr: curr + ind_batch_size],
                                            model.intents: ind_valid_data['intent'][curr: curr + ind_batch_size],
                                            model.utter_len: ind_valid_data['len'][curr: curr + ind_batch_size],
                                            model.keep_rate: 1.0})
                for i in range(len(ind_valid_data['intent'][curr: curr + ind_batch_size])):
                    f.write(' '.join([str(j) for j in [ind_valid_data['intent'][curr + i]] + enc_h[i].tolist()]) + '\n')
                curr += ind_batch_size

        ood_count = len(ood_valid_data['intent'])
        logger.info('generating {} ood samples to {}'.format(ood_count, ood_outfile))
        with open(ood_outfile, 'w') as f:
            curr = 0
            while curr < ood_count:
                enc_h = sess.run(model.enc_h,
                                 feed_dict={model.utter: ood_valid_data['utter'][curr: curr + ind_batch_size],
                                            model.intents: ood_valid_data['intent'][curr: curr + ind_batch_size],
                                            model.utter_len: ood_valid_data['len'][curr: curr + ind_batch_size],
                                            model.keep_rate: 1.0})
                for i in range(len(ood_valid_data['intent'][curr: curr + ind_batch_size])):
                    f.write(' '.join([str(j) for j in enc_h[i].tolist()]) + '\n')
                curr += ind_batch_size

        logger.info('generating {} gen samples to {}'.format(gen_count, gen_outfile))
        with open(gen_outfile, 'w') as f:
            curr = 0
            while curr < gen_count:
                gen_out = sess.run(model.generator_out,
                                   feed_dict={model.batch_size: gen_batch_size, model.keep_rate: 1.0})
                for i in range(gen_batch_size):
                    f.write(' '.join([str(j) for j in gen_out[i].tolist()]) + '\n')
                curr += ind_batch_size

        logger.info('fin.')

except:
    logger.error(traceback.format_exc())




