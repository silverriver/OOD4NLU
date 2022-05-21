import tensorflow as tf
import utils
import argparse
from model import ICM
import os
import time
import traceback
import numpy as np
import pickle
from data_helper import TFRData
from gen_tfrecord import gen_tfrecord
import eval

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    help='config file',
                    default='config.json')
parser.add_argument('--gpu', help='which gpu to use', default='2')
parser.add_argument("--is_train", type=utils.str2bool, default=True, help="is_train or infer&evaluate")

args = parser.parse_args()

config = utils.load_config(args.config)
base_dir = os.path.dirname(args.config)
logger = utils.get_logger(os.path.join(base_dir, 'main.log'))

ind_train_data = os.path.join(base_dir, config['ind_train_data'])
ood_train_data = os.path.join(base_dir, config['ood_train_data'])
ind_valid_data = os.path.join(base_dir, config['ind_valid_data'])
ood_valid_data = os.path.join(base_dir, config['ood_valid_data'])
ind_test_data = os.path.join(base_dir, config['ind_test_data'])
ood_test_data = os.path.join(base_dir, config['ood_test_data'])
train_dir = os.path.join(base_dir, config['train_dir'])
data_dir = os.path.join(base_dir, config['data_dir'])
eval_dir = os.path.join(base_dir, config['eval_dir'])
log_dir = os.path.join(base_dir, config['log_dir'])
preprocess_dir = os.path.join(base_dir, config['preprocess_dir'])
best_model = os.path.join(base_dir, config['best_model'])
char_vocab_file = os.path.join(preprocess_dir, config['char_vocab_file'])
intent_vocab_file = os.path.join(preprocess_dir, config['intent_vocab_file'])
ind_train_prep_file = os.path.join(preprocess_dir, config['ind_train_prep_file'])
ood_train_prep_file = os.path.join(preprocess_dir, config['ood_train_prep_file'])
ind_valid_prep_file = os.path.join(preprocess_dir, config['ind_valid_prep_file'])
ood_valid_prep_file = os.path.join(preprocess_dir, config['ood_valid_prep_file'])
ind_test_prep_file = os.path.join(preprocess_dir, config['ind_test_prep_file'])
ood_test_prep_file = os.path.join(preprocess_dir, config['ood_test_prep_file'])


def add_summary(writer, step, data):
    for i in data:
        name, value = i
        writer.add_summary(summary=tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)]), global_step=step)


np.random.seed(config['seed'])

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info('tf Version: {}'.format(tf.__version__))
    for i in config:
        logger.info('{}: {}'.format(i, config[i]))
    for i in vars(args):
        logger.info('{}: {}'.format(i, getattr(args, i)))

    dirs = [train_dir, eval_dir, log_dir, preprocess_dir, best_model]
    for d in dirs:
        if not os.path.isdir(d):
            logger.info('cannot find {}, mkdiring'.format(d))
            os.makedirs(d)

    if args.is_train:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            check_list = {'char': char_vocab_file, 'intent': intent_vocab_file}
            if tf.train.get_checkpoint_state(train_dir):
                # load trained model
                logger.info('Found check point in {}'.format(train_dir))
                for file in check_list:
                    if not os.path.isfile(check_list[file]):
                        logger.error('Can not find {}'.format(check_list[file]))
                        exit(1)
                logger.info('Loading vocab files {}'.format(tf.train.latest_checkpoint(train_dir)))
                char2id, id2char = utils.read_vocab(check_list['char'], logger, limit=config['char_vocab_size'])
                intent2id, id2intent = utils.read_vocab(check_list['intent'], logger)

                config = utils.update_vocab_size(config, intent2id)
                logger.info('Loading checkpoint from {}'.format(tf.train.latest_checkpoint(train_dir)))
                model = ICM(config)
                model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
            else:
                # Begin from scratch
                logger.info('Starting from scratch')
                miss_files = [file for file in check_list.values() if not os.path.isfile(file)]
                if len(miss_files) != 0:
                    logger.info('{} files '.format(len(miss_files)) +
                                'are missing: \n {}'.format('\n'.join(miss_files)))
                    logger.info('re-building vocab')
                    utils.build_vocab([ind_train_data, ood_train_data, ood_valid_data, ind_valid_data, ood_test_data,
                                       ind_test_data], check_list['char'], check_list['intent'], logger)

                logger.info('Reading vocab files')
                char2id, id2char = utils.read_vocab(check_list['char'], logger, config['char_vocab_size'])
                intent2id, id2intent = utils.read_vocab(check_list['intent'], logger)

                config = utils.update_vocab_size(config, intent2id)
                embed = utils.load_embed(char2id, config['char_vocab_size'], config['char_embed_size'],
                                         config['pretrained_embed'], logger)
                model = ICM(config, embed)
                sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
            valid_writer = tf.summary.FileWriter(os.path.join(log_dir, 'valid'), sess.graph)

            best_saver = tf.train.Saver(max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

            train_intent_loss = 0.0
            train_kl_loss = 0.0
            train_acc = 0.0
            train_time = 0.0
            best_acc = 0.0
            last_improved = 0
            start_time = time.time()

            prev_loss = [1e18 for _ in range(5)]

            logger.info('Preparing data')
            if not os.path.isfile(ind_train_prep_file):
                logger.info('IND Train file not preprocessed, processing')
                gen_tfrecord(char2id, intent2id, [ind_train_data], [ind_train_prep_file],
                             logger, config['max_l'], shuffle=True)

            if not os.path.isfile(ood_train_prep_file):
                logger.info('OOD Train file not preprocessed, processing')
                gen_tfrecord(char2id, intent2id, [ood_train_data], [ood_train_prep_file],
                             logger, config['max_l'], shuffle=True)

            if not os.path.isfile(ind_valid_prep_file):
                logger.info('IND valid file not preprocessed, processing')
                gen_tfrecord(char2id, intent2id, [ind_valid_data], [ind_valid_prep_file],
                             logger, config['max_l'], shuffle=False)

            if not os.path.isfile(ood_valid_prep_file):
                logger.info('OOD valid file not preprocessed, processing')
                gen_tfrecord(char2id, intent2id, [ood_valid_data], [ood_valid_prep_file],
                             logger, config['max_l'], shuffle=False)

            logger.info('Loading train data')
            ind_data_train = TFRData(char2id[utils._PAD], seed=config['seed'],
                                 shuffle_buffer=1000, prefetch=1000, repeat=config['max_epoch'])
            ind_data_train.init(sess, [ind_train_prep_file], config['batch_size'])
            ind_data_train_handler = ind_data_train.get_handler(sess)

            ood_data_train = TFRData(char2id[utils._PAD], seed=config['seed'],
                                     shuffle_buffer=1000, prefetch=1000, repeat=True)
            ood_data_train.init(sess, [ood_train_prep_file], config['batch_size'])
            ood_data_train_handler = ood_data_train.get_handler(sess)

            logger.info('Loading valid data')
            ind_data_valid = TFRData(char2id[utils._PAD])
            ind_data_valid_handler = ind_data_valid.get_handler(sess)

            ood_data_valid = TFRData(char2id[utils._PAD])
            ood_data_valid_handler = ood_data_valid.get_handler(sess)

            logger.info('Training')
            while True:
                try:
                    start_time = time.time()
                    _, _, summ, intent_loss, kl_loss, acc = sess.run([
                        model.ind_train_op, model.ood_train_op, model.summ_op, model.intent_loss, model.kl_loss, model.acc],
                        feed_dict={model.ind_data_handler: ind_data_train_handler,
                                   model.ood_data_handler: ood_data_train_handler, model.keep_rate: config['keep_rate']})

                    train_intent_loss += intent_loss
                    train_kl_loss += kl_loss
                    train_acc += acc
                    train_time += (time.time() - start_time)

                    total_step = model.ind_step.eval(sess)

                    if total_step % config['save_per_iter'] == 0:
                        # save
                        model.saver.save(sess, os.path.join(train_dir, 'model.ckpt'), global_step=total_step)

                        train_intent_loss /= config['save_per_iter']
                        train_kl_loss /= config['save_per_iter']
                        train_acc /= config['save_per_iter']
                        train_time /= config['save_per_iter']

                        learning_rate = model.lr.eval(sess)

                        train_writer.add_summary(summ, total_step)

                        if train_intent_loss + train_kl_loss > max(prev_loss):
                            sess.run(model.lr_decay_op)
                        prev_loss = prev_loss[1:] + [train_intent_loss + train_kl_loss]

                        ind_data_valid.init(sess, [ind_valid_prep_file], config['batch_size'])
                        ood_data_valid.init(sess, [ood_valid_prep_file], config['batch_size'])
                        valid_intent_loss, valid_kl_loss, valid_acc, valid_bs = 0.0, 0.0, 0.0, 0.0
                        ind_softmax = []
                        ood_softmax = []
                        while True:
                            try:
                                intent_loss, acc, ind_bs, softmax = sess.run(
                                    [model.intent_loss, model.acc, model.ind_bs, model.intent_softmax],
                                    feed_dict={model.ind_data_handler: ind_data_valid_handler, model.keep_rate: 1.0})

                                valid_intent_loss += intent_loss * ind_bs
                                valid_acc += acc * ind_bs
                                valid_bs += ind_bs
                                ind_softmax.append(softmax)
                            except tf.errors.OutOfRangeError:
                                break
                        valid_intent_loss /= valid_bs
                        valid_acc /= valid_bs
                        ind_softmax = np.concatenate(ind_softmax, axis=0)

                        valid_bs = 0.0
                        while True:
                            try:
                                kl_loss, softmax, ood_bs = sess.run(
                                    [model.kl_loss, model.ood_intent_softmax, model.ood_bs],
                                    feed_dict={model.ood_data_handler: ood_data_valid_handler, model.keep_rate: 1.0})

                                valid_kl_loss = kl_loss * ood_bs
                                valid_bs += ood_bs
                                ood_softmax.append(softmax)
                            except tf.errors.OutOfRangeError:
                                break
                        valid_kl_loss /= valid_bs
                        ood_softmax = np.concatenate(ood_softmax, axis=0)
                        auroc = eval.cal_roc_auc(ind_softmax, ood_softmax)
                        fpr_at_tpr95 = eval.fpr_at_tpr95(ind_softmax, ood_softmax)[0]

                        add_summary(valid_writer, total_step, data=[('acc', valid_acc), ('intent_loss', valid_intent_loss),
                                                                    ('kl_loss', valid_kl_loss),
                                                                    ('auroc', auroc), ('fpr_at_tpr95', fpr_at_tpr95)])

                        if valid_acc > best_acc:
                            best_acc = valid_acc
                            best_saver.save(sess=sess, save_path=os.path.join(best_model, 'best_model.ckpt'),
                                            global_step=total_step)
                            last_improved = total_step
                            improved_str = '*'
                        else:
                            improved_str = ' '

                        format_str = 'It: {0:>5} t intent loss: {1:>4.4f} t kl loss: {2:>4.4f} t acc: {3:>2.2%} ' + \
                                     'v intent loss: {4:>4.4f} v kl loss: {5:>4.4f} v acc: {6:>2.2%}, auroc: {7:>4.4f}, fpr@tpr95: {8:>4.4f}, ' +\
                                     't: {9:>3.4f}ms lr: {10:>.6} {11}'
                        out1 = format_str.format(total_step, train_intent_loss, train_kl_loss, train_acc,
                                                 valid_intent_loss, valid_kl_loss, valid_acc, auroc, fpr_at_tpr95, train_time * 1000,
                                                 learning_rate, improved_str)

                        logger.info(out1)

                        train_intent_loss = 0
                        train_kl_loss = 0
                        train_acc = 0
                        train_time = 0

                except tf.errors.OutOfRangeError:
                    logger.info('Training complete')
                    break
    else:
        ckpt_file = os.path.join(best_model, 'checkpoint')
        if not os.path.isfile(ckpt_file):
            logger.error("cannot find {}".format(ckpt_file))
            exit(1)

        with open(ckpt_file) as f:
            model_ckpts = [i.strip() for i in f.readlines() if 'all_model_checkpoint_paths' in i]
        model_ckpts = [i.split(":")[1].strip()[1:][:-1] for i in model_ckpts]
        logger.info("found {} checkpoints".format(len(model_ckpts)))
        for model_path in model_ckpts:
            if not tf.train.checkpoint_exists(model_path):
                logger.error('cannot find model {}'.format(model_path))
                exit(1)
            else:
                logger.info("{}".format(model_path))

        prev_res = [i for i in os.listdir(eval_dir) if '.pkl' in i]
        logger.info('found {} previous files, deleting'.format(len(prev_res)))
        for prev in prev_res:
            logger.info(prev)
            os.remove(os.path.join(eval_dir, prev))

        char2id, id2char = utils.read_vocab(char_vocab_file, logger, config['char_vocab_size'])
        intent2id, id2intent = utils.read_vocab(intent_vocab_file, logger)

        logger.info('Processing test ind data')
        gen_tfrecord(char2id, intent2id, [ind_test_data], [ind_test_prep_file],
                     logger, config['max_l'], word_level=True, shuffle=False)

        logger.info('Processing test ood data')
        gen_tfrecord(char2id, intent2id, [ood_test_data], [ood_test_prep_file],
                     logger, config['max_l'], word_level=True, shuffle=False)

        logger.info('Processing valid ind data')
        gen_tfrecord(char2id, intent2id, [ind_valid_data], [ind_valid_prep_file],
                     logger, config['max_l'], word_level=True, shuffle=False)

        logger.info('Processing valid ood data')
        gen_tfrecord(char2id, intent2id, [ood_valid_data], [ood_valid_prep_file],
                     logger, config['max_l'], word_level=True, shuffle=False)

        test_loss_list = []
        test_ood_loss_list = []
        test_acc_list = []
        test_auc_list = []
        test_fpr_at_tpr95_list = []
        test_fpr_at_tpr90_list = []

        for model_path in model_ckpts:
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            tf.reset_default_graph()
            with tf.Session(config=tf_config) as sess:
                logger.info('evaling model {}'.format(model_path))
                for file in [ind_test_data, ind_test_prep_file]:
                    if not os.path.isfile(file):
                        logger.error('cannot find test file {}'.format(file))
                        exit(1)

                config = utils.update_vocab_size(config, intent2id)
                logger.info('loading checkpoint from {}'.format(model_path))
                model = ICM(config)
                model.saver.restore(sess, model_path)

                logger.info('Loading test data')

                ind_data_test = TFRData(char2id[utils._PAD])
                ood_data_test = TFRData(char2id[utils._PAD])
                ind_data_test_handler = ind_data_test.get_handler(sess)
                ood_data_test_handler = ood_data_test.get_handler(sess)

                logger.info('evaluating test file')

                ind_data_test.init(sess, [ind_test_prep_file], config['batch_size'])
                ood_data_test.init(sess, [ood_test_prep_file], config['batch_size'])

                test_loss, test_acc, test_bs = 0.0, 0.0, 0.0
                ind_res = {'softmax': [], 'gt': [], 'logit': [], 'id': []}
                ood_res = {'softmax': [], 'gt': [], 'logit': [], 'id': []}
                while True:
                    try:
                        loss, acc, bs, softmax, gt, logit, id = sess.run(
                            [model.intent_loss, model.acc, model.ind_bs, model.intent_softmax, model.ind_intent,
                             model.ind_logits, model.ind_ids],
                            feed_dict={model.ind_data_handler: ind_data_test_handler, model.keep_rate: 1.0})

                        test_loss += loss * bs
                        test_acc += acc * bs
                        test_bs += bs
                        ind_res['softmax'].append(softmax)
                        ind_res['gt'].append(gt)
                        ind_res['logit'].append(logit)
                        ind_res['id'].append(id)
                    except tf.errors.OutOfRangeError:
                        break
                test_loss /= test_bs
                test_acc /= test_bs

                ind_res['softmax'] = np.concatenate(ind_res['softmax'], axis=0)
                ind_res['gt'] = np.concatenate(ind_res['gt'], axis=0)
                ind_res['logit'] = np.concatenate(ind_res['logit'], axis=0)
                ind_res['id'] = np.concatenate(ind_res['id'], axis=0)

                # test_loss, test_bs = 0.0, 0.0
                while True:
                    try:
                        loss, bs, softmax, gt, logit, id = sess.run(
                            [model.kl_loss, model.ood_bs, model.ood_intent_softmax, model.ood_intent, model.ood_logits, model.ood_ids],
                            feed_dict={model.ood_data_handler: ood_data_test_handler, model.keep_rate: 1.0})

                        # test_loss += loss * bs
                        test_bs += bs
                        ood_res['softmax'].append(softmax)
                        ood_res['gt'].append(gt)
                        ood_res['logit'].append(logit)
                        ood_res['id'].append(id)
                    except tf.errors.OutOfRangeError:
                        break

                ood_res['softmax'] = np.concatenate(ood_res['softmax'], axis=0)
                ood_res['gt'] = np.concatenate(ood_res['gt'], axis=0)
                ood_res['logit'] = np.concatenate(ood_res['logit'], axis=0)
                ood_res['id'] = np.concatenate(ood_res['id'], axis=0)

                auroc = eval.cal_roc_auc(ind_res['softmax'], ood_res['softmax'])
                fpr_at_tpr95 = eval.fpr_at_tpr95(ind_res['softmax'], ood_res['softmax'])[0]
                fpr_at_tpr90 = eval.fpr_at_tprN(ind_res['softmax'], ood_res['softmax'], 0.9)[0]

                test_loss_list.append(test_loss)
                test_acc_list.append(test_acc)
                test_auc_list.append(auroc)
                test_fpr_at_tpr95_list.append(fpr_at_tpr95)
                test_fpr_at_tpr90_list.append(fpr_at_tpr90)

                logger.info('test_loss: {}, test_acc: {}, auroc: {}, fpr@tpr95: {}, fpr@tpr90'.format(
                    test_loss, test_acc, auroc, fpr_at_tpr95, fpr_at_tpr90))

                test_ind_res_file = os.path.join(eval_dir, os.path.basename(model_path) + '_test_ind_res.pkl')
                test_ood_res_file = os.path.join(eval_dir, os.path.basename(model_path) + '_test_ood_res.pkl')
                with open(test_ind_res_file, 'wb') as file:
                    pickle.dump(ind_res, file)

                with open(test_ood_res_file, 'wb') as file:
                    pickle.dump(ood_res, file)

                logger.info('test res saved to \n{}\n {}'.format(test_ind_res_file, test_ood_res_file))

                logger.info('evaluating valid file')

                ind_data_test.init(sess, [ind_valid_prep_file], config['batch_size'])
                ood_data_test.init(sess, [ood_valid_prep_file], config['batch_size'])
                ind_res = {'softmax': [], 'gt': [], 'logit': [], 'id': []}
                ood_res = {'softmax': [], 'gt': [], 'logit': [], 'id': []}
                while True:
                    try:
                        softmax, gt, logit, id = sess.run(
                            [model.intent_softmax, model.ind_intent, model.ind_logits, model.ind_ids],
                            feed_dict={model.ind_data_handler: ind_data_test_handler, model.keep_rate: 1.0})
                        ind_res['softmax'].append(softmax)
                        ind_res['gt'].append(gt)
                        ind_res['logit'].append(logit)
                        ind_res['id'].append(id)
                    except tf.errors.OutOfRangeError:
                        break
                ind_res['softmax'] = np.concatenate(ind_res['softmax'], axis=0)
                ind_res['gt'] = np.concatenate(ind_res['gt'], axis=0)
                ind_res['logit'] = np.concatenate(ind_res['logit'], axis=0)
                ind_res['id'] = np.concatenate(ind_res['id'], axis=0)

                while True:
                    try:
                        softmax, gt, logit, id = sess.run(
                            [model.ood_intent_softmax, model.ood_intent, model.ood_logits, model.ood_ids],
                            feed_dict={model.ood_data_handler: ood_data_test_handler, model.keep_rate: 1.0})
                        ood_res['softmax'].append(softmax)
                        ood_res['gt'].append(gt)
                        ood_res['logit'].append(logit)
                        ood_res['id'].append(id)
                    except tf.errors.OutOfRangeError:
                        break
                ood_res['softmax'] = np.concatenate(ood_res['softmax'], axis=0)
                ood_res['gt'] = np.concatenate(ood_res['gt'], axis=0)
                ood_res['logit'] = np.concatenate(ood_res['logit'], axis=0)
                ood_res['id'] = np.concatenate(ood_res['id'], axis=0)

                valid_ind_res_file = os.path.join(eval_dir, os.path.basename(model_path) + '_valid_ind_res.pkl')
                valid_ood_res_file = os.path.join(eval_dir, os.path.basename(model_path) + '_valid_ood_res.pkl')
                with open(valid_ind_res_file, 'wb') as file:
                    pickle.dump(ind_res, file)

                with open(valid_ood_res_file, 'wb') as file:
                    pickle.dump(ood_res, file)

                logger.info('valid res saved to \n{}\n {}'.format(valid_ind_res_file, valid_ood_res_file))

        logger.info("========Final Performance==========")
        logger.info('test_loss: {}'.format(utils.mean_var(test_loss_list)))
        logger.info('test_acc: {}'.format(utils.mean_var(test_acc_list)))
        logger.info('auc: {}'.format(utils.mean_var(test_auc_list)))
        logger.info('fpr_at_tpr95: {}'.format(utils.mean_var(test_fpr_at_tpr95_list)))
        logger.info('fpr_at_tpr90: {}'.format(utils.mean_var(test_fpr_at_tpr90_list)))
except:
    logger.error(traceback.format_exc())




