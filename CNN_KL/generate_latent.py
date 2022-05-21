import tensorflow as tf
import utils
import argparse
from model import ICM
import os
import traceback
import gen_tfrecord
import numpy as np
from data_helper import TFRData

parser = argparse.ArgumentParser()

parser.add_argument('--config', help='config file',
                    default='config.json')
parser.add_argument('--gpu', help='which gpu to use', default='3')
parser.add_argument('--ind_outfile', help='path to the output file of ind', default='/tmp/ind_outfile')
parser.add_argument('--ood_outfile', help='path to the output file of ood', default='/tmp/ood_outfile')

args = parser.parse_args()
config = utils.load_config(args.config)

poj_base = os.path.dirname(args.config)
logger = utils.get_logger(os.path.join(poj_base, 'main.log'))
valid_file_ind = os.path.join(poj_base, config['ind_test_data'])
valid_file_ood = os.path.join(poj_base, config['ood_test_data'])
data_dir = os.path.join(poj_base, config['data_dir'])
eval_dir = os.path.join(poj_base, config['eval_dir'])
preprocess_dir = os.path.join(poj_base, config['preprocess_dir'])
best_model = os.path.join(poj_base, config['best_model'])
char_vocab_file = os.path.join(preprocess_dir, config['char_vocab_file'])
intent_vocab_file = os.path.join(preprocess_dir, config['intent_vocab_file'])
valid_prep_file_ind = os.path.join(preprocess_dir, config['ind_valid_prep_file'])
valid_prep_file_ood = os.path.join(preprocess_dir, config['ood_valid_prep_file'])


ind_outfile = args.ind_outfile
ood_outfile = args.ood_outfile
np.random.seed(config['seed'])


try:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info('tf Version: {}'.format(tf.__version__))
    for i in config:
        logger.info('{}: {}'.format(i, config[i]))
    for i in vars(args):
        logger.info('{}: {}'.format(i, getattr(args, i)))

    print('best_model', best_model)
    ckpt_file = tf.train.latest_checkpoint(best_model)
    if not ckpt_file:
        logger.error("cannot find checkpoint in {}".format(best_model))
        exit(1)

    logger.info("found checkpoint {}".format(ckpt_file))

    char2id, id2char = utils.read_vocab(char_vocab_file, logger, config['char_vocab_size'])
    intent2id, id2intent = utils.read_vocab(intent_vocab_file, logger)
    config = utils.update_vocab_size(config, intent2id)

    logger.info('Processing valid ind data')
    gen_tfrecord.gen_tfrecord(char2id, intent2id, [valid_file_ind], [valid_prep_file_ind],
                              logger, config['max_l'], word_level=True, shuffle=False)

    logger.info('Processing valid ood data')
    gen_tfrecord.gen_tfrecord(char2id, intent2id, [valid_file_ood], [valid_prep_file_ood],
                              logger, config['max_l'], word_level=True, shuffle=False)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Session(config=tf_config) as sess:
        logger.info('loading model from {}'.format(ckpt_file))
        model = ICM(config=config)
        model.saver.restore(sess, ckpt_file)

        logger.info('Loading test data')

        ind_data_valid = TFRData(char2id[utils._PAD])
        ood_data_valid = TFRData(char2id[utils._PAD])
        ind_data_valid_handler = ind_data_valid.get_handler(sess)
        ood_data_valid_handler = ood_data_valid.get_handler(sess)

        ind_data_valid.init(sess, [valid_prep_file_ind], config['batch_size'])
        ood_data_valid.init(sess, [valid_prep_file_ood], config['batch_size'])

        logger.info('generating ind samples to {}'.format(ind_outfile))
        with open(ind_outfile, 'w') as f:
            while True:
                try:
                    features, intent = sess.run([model.ind_hidden_features, model.ind_intent],
                                                feed_dict={model.ind_data_handler: ind_data_valid_handler, model.keep_rate: 1.0})
                    for i in range(intent.shape[0]):
                        f.write(' '.join([str(j) for j in [intent[i]] + features[i].tolist()]) + '\n')
                except tf.errors.OutOfRangeError:
                    break
        logger.info('generating ood samples to {}'.format(ood_outfile))
        with open(ood_outfile, 'w') as f:
            while True:
                try:
                    features = sess.run(model.ood_hidden_features,
                                        feed_dict={model.ood_data_handler: ood_data_valid_handler, model.keep_rate: 1.0})
                    for i in range(features.shape[0]):
                        f.write(' '.join([str(j) for j in features[i].tolist()]) + '\n')
                except tf .errors.OutOfRangeError:
                    break

        logger.info('fin.')

except:
    logger.error(traceback.format_exc())




