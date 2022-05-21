import tensorflow as tf
import utils
import argparse
from model import CGAN
from cls_model import ICM
import os
import traceback
import numpy as np
import random


def output_infer(cgan_model, cgan_sess, cgan_id2word, cgan_id2intent, cls_char2id, cls_intent2id, cgan_intent):
    greedy_infer = cgan_sess.run(cgan_model.latent_greedy_utter,
                                 feed_dict={cgan_model.intent_sample: cgan_intent, cgan_model.keep_rate: 1.0,
                                            cgan_model.batch_size: cgan_intent.shape[0]})
    cls_utter = []
    cls_intent = np.zeros_like(cgan_intent)
    for i in range(greedy_infer.shape[0]):
        utter = greedy_infer[i]
        eos_pos = utils._find_eos(utter, utils.EOS_ID)
        utter = utter[:eos_pos]
        utter = ''.join(cgan_id2word[j] for j in utter)
        utter.replace(' ', '')
        cls_utter.append([cls_char2id[j] if j in cls_char2id else cls_char2id[utils._UNK] for j in utter] +
                         [cls_char2id[utils._EOS]])
    max_len = max([len(i) for i in cls_utter])
    for i in range(len(cls_utter)):
        cls_utter[i] = cls_utter[i] + [cls_char2id[utils._PAD]] * (max_len - len(cls_utter[i]))
    cls_utter = np.asarray(cls_utter, dtype=np.float32)
    for i in range(cgan_intent.shape[0]):
        cls_intent[i] = cls_intent2id[cgan_id2intent[cgan_intent[i]]]

    return cls_utter, cls_intent


logger = utils.get_logger('interp.log')
parser = argparse.ArgumentParser()

parser.add_argument('--cgan_config', help='config file', default='config.json')
parser.add_argument('--cls_config', help='config file',
                    default='config.json')
parser.add_argument('--gpu', help='which gpu to use', default='0')
parser.add_argument("--batch_size", type=int, default=500, help="how many instance to test")
parser.add_argument("--batch_count", type=int, default=1000, help="how many instance to test")

args = parser.parse_args()
config = utils.load_config(args.cgan_config)
cls_config = utils.load_config(args.cls_config)

cls_best_dir = os.path.join(cls_config['poj_base'], cls_config['best_model'])
cls_char_vocab = os.path.join(cls_config['poj_base'], cls_config['preprocess_dir'], cls_config['char_vocab_file'])
cls_intent_vocab = os.path.join(cls_config['poj_base'], cls_config['preprocess_dir'], cls_config['intent_vocab_file'])

train_dir = os.path.join(config['poj_base'], config['train_dir'])
preprocess_dir = os.path.join(config['poj_base'], config['preprocess_dir'])
word_vocab_file = os.path.join(preprocess_dir, config['word_vocab_file'])
intent_vocab_file = os.path.join(preprocess_dir, config['intent_vocab_file'])
batch_count = args.batch_count
batch_size = args.batch_size


try:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info('tf Version: {}'.format(tf.__version__))
    for i in config:
        logger.info('{}: {}'.format(i, config[i]))
    for i in vars(args):
        logger.info('{}: {}'.format(i, getattr(args, i)))

    ckpt_dir = train_dir
    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    if not ckpt_file:
        logger.error("cannot find checkpoint in {}".format(ckpt_dir))
        exit(1)

    cls_ckpt_file = tf.train.latest_checkpoint(cls_best_dir)
    if not cls_ckpt_file:
        logger.error("cannot find checkpoint in {}".format(ckpt_dir))
        exit(1)

    logger.info("found cgan checkpoint {}".format(ckpt_file))
    logger.info("found cls checkpoint {}".format(cls_ckpt_file))

    word2id, id2word = utils.read_vocab(word_vocab_file, logger, config['word_vocab_size'])
    cgan_intent2id, cgan_id2intent = utils.read_vocab(intent_vocab_file, logger)
    cls_intent2id, cls_id2intent = utils.read_vocab(intent_vocab_file, logger)
    char2id, id2char = utils.read_vocab(cls_char_vocab, logger, cls_config['char_vocab_size'])
    config = utils.update_vocab_size(config, cgan_intent2id, dict())
    cls_config = utils.update_vocab_size(cls_config, cls_intent2id, dict())

    cls_g = tf.Graph()
    cgan_g = tf.Graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    cls_sess = tf.Session(config=tf_config, graph=cls_g)
    cgan_sess = tf.Session(config=tf_config, graph=cgan_g)
    with cgan_g.as_default():
        logger.info('loading cgan model from {}'.format(ckpt_file))
        cgan_model = CGAN(config, utils.GO_ID, utils.EOS_ID)
        cgan_model.saver.restore(cgan_sess, ckpt_file)

    with cls_g.as_default():
        logger.info('loading cls model from {}'.format(cls_ckpt_file))
        cls_config = utils.update_vocab_size(cls_config, cls_intent2id, dict())
        cls_model = ICM(cls_config)
        cls_model.saver.restore(cls_sess, cls_ckpt_file)

    sample_intent = np.random.randint(0, config['intent_num'] - 1, size=[batch_count * batch_size], dtype=np.int32)
    pred_intent = np.zeros_like(sample_intent)
    for i in range(batch_count):
        if i % 100 == 0:
            print(i)

        start = i * batch_size
        end = (i + 1) * batch_size

        # rand_intent = random.choice(intent_pair)
        cls_utter, cls_intent = output_infer(cgan_model, cgan_sess, id2word, cgan_id2intent,
                                             char2id, cls_intent2id, sample_intent[start: end])

        intent_pred, intent_sm = cls_sess.run([cls_model.intent_predict, cls_model.intent_softmax],
                                              feed_dict={cls_model.ind_utt: cls_utter, cls_model.keep_rate: 1.0})

        for j in range(batch_size):
            pred_intent[start + j] = cls_intent2id[cgan_id2intent[intent_pred[j]]]
    print('pred acc.: {}'.format(np.mean(np.equal(pred_intent, sample_intent))))

except:
    logger.error(traceback.format_exc())




