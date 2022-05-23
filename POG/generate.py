import tensorflow as tf
import utils
import argparse
from model import CGAN
import os
import time
import traceback
import numpy as np
from data_helper import TFRData
from gen_tfrecord import gen_tfrecord

parser = argparse.ArgumentParser()

parser.add_argument('--config', help='config file', default='config.json')
parser.add_argument('--gpu', help='which gpu to use', default='1')
parser.add_argument('--outfile', help='path to the output file', default='/tmp/outfile')
parser.add_argument('--count', help='how many data to generate', type=int, default=50000)
parser.add_argument('--batch_size', help='batch_size', type=int, default=500)
parser.add_argument('--is_sample', help='use sample or not (greedy)', type=utils.str2bool, default=False)
parser.add_argument('--sample_t', help='sample temperature', type=float, default=1.0)

args = parser.parse_args()
config = utils.load_config(args.config)

poj_base = os.path.dirname(args.config)
train_dir = os.path.join(poj_base, config['train_dir'])
preprocess_dir = os.path.join(poj_base, config['preprocess_dir'])

word_vocab_file = os.path.join(preprocess_dir, config['word_vocab_file'])
intent_vocab_file = os.path.join(preprocess_dir, config['intent_vocab_file'])
outfile = args.outfile
count = args.count
batch_size = args.batch_size
is_sample = args.is_sample
sample_t = args.sample_t

logger = utils.get_logger(os.path.join(poj_base, 'main.log'))


def log_batch(infer, id2word, logger):
    bs = infer.shape[0]
    for i in range(bs):
        utter = infer[i]
        eos_pos = utils._find_eos(utter, utils.EOS_ID)
        logger.info('{}: {}'.format(i, ' '.join([id2word[j] for j in utter[:eos_pos]])))


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
    config['word_vocab_size'] = min(config['word_vocab_size'], len(word2id)) 
    intent2id, id2intent = utils.read_vocab(intent_vocab_file, logger)
    config = utils.update_vocab_size(config, intent2id, dict())

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
        p_logger.info("---------noise sample--------------")
        noise_sample = sess.run(model.noise_greedy_utter,
                                feed_dict={model.batch_size: 5, model.keep_rate: 1.0})
        log_batch(noise_sample, id2word, p_logger)
        p_logger.info("---------latent sample--------------")
        sample_latent = sess.run(model.latent_sample_utter,
                                 feed_dict={model.batch_size: 5, model.keep_rate: 1.0, model.sample_t: sample_t})
        log_batch(sample_latent, id2word, p_logger)
        p_logger.info("---------latent greedy--------------")
        greedy_latent = sess.run(model.latent_greedy_utter,
                                 feed_dict={model.batch_size: 5, model.keep_rate: 1.0})
        log_batch(greedy_latent, id2word, p_logger)

        logger.info('generating {} samples to {}'.format(count, outfile))
        with open(outfile, 'w') as f:
            curr = 0
            while curr < count:
                if is_sample:
                    utter = sess.run(model.latent_sample_utter,
                                     feed_dict={model.batch_size: batch_size, model.keep_rate: 1.0,
                                                model.sample_t: sample_t})
                else:
                    utter = sess.run(model.latent_greedy_utter,
                                     feed_dict={model.batch_size: batch_size, model.keep_rate: 1.0})
                for i in range(utter.shape[0]):
                    utt = utter[i]
                    eos_pos = utils._find_eos(utt, utils.EOS_ID)
                    if eos_pos != 0:
                        f.write('ood\t{}\n'.format(' '.join(id2word[j] for j in utt[:eos_pos])))
                curr += batch_size

        logger.info('fin.')

except:
    logger.error(traceback.format_exc())




