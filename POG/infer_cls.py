import tensorflow as tf
import utils
import argparse
from model import CGAN
from gen_tfrecord import load_data
import os
import traceback
import numpy as np
import sklearn


logger = utils.get_logger('interp.log')
parser = argparse.ArgumentParser()

parser.add_argument('--cgan_config', help='config file',
                    default='config.json')
parser.add_argument('--infile', help='input file', default='data/dl_root_ind_test')
parser.add_argument('--gpu', help='which gpu to use', default='0')
parser.add_argument("--batch_size", type=int, default=500, help="how many instance to test")

args = parser.parse_args()
config = utils.load_config(args.cgan_config)

train_dir = os.path.join(config['poj_base'], config['train_dir'])
preprocess_dir = os.path.join(config['poj_base'], config['preprocess_dir'])
word_vocab_file = os.path.join(preprocess_dir, config['word_vocab_file'])
intent_vocab_file = os.path.join(preprocess_dir, config['intent_vocab_file'])
batch_size = args.batch_size
infile = args.infile


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

    logger.info("found cgan checkpoint {}".format(ckpt_file))

    word2id, id2word = utils.read_vocab(word_vocab_file, logger, config['word_vocab_size'])
    intent2id, id2intent = utils.read_vocab(intent_vocab_file, logger)
    config = utils.update_vocab_size(config, intent2id, dict())

    cgan_g = tf.Graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config, graph=cgan_g)
    with cgan_g.as_default():
        logger.info('loading cgan model from {}'.format(ckpt_file))
        cgan_model = CGAN(config, utils.GO_ID, utils.EOS_ID)
        cgan_model.saver.restore(sess, ckpt_file)

    test_data = load_data(word2id, intent2id, [infile], logger, config['max_utter_len'], word_level=True, shuffle=True)
    data_size = len(test_data['intent'])
    pred = []
    count = 0
    logger.info('test, data_size: {}'.format(data_size))
    while count < data_size:
        cls_infer_pred = sess.run(cgan_model.cls_infer_pred, feed_dict={
            cgan_model.utter: test_data['utter'][count:count + batch_size],
            cgan_model.utter_len: test_data['len'][count:count + batch_size],
            cgan_model.keep_rate: 1.0})
        pred = np.concatenate((pred, cls_infer_pred))
        count += batch_size

    print('acc.: {}'.format(sklearn.metrics.accuracy_score(test_data['intent'], pred)))
    print('f1.: {}'.format(sklearn.metrics.f1_score(test_data['intent'], pred, average='micro')))


except:
    logger.error(traceback.format_exc())
