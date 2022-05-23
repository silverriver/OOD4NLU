import tensorflow as tf
import utils
import argparse
from model import CGAN
import os
import time
import traceback
import pickle
import numpy as np
import random
import eval
import sklearn
from data_helper import TFRData
from gen_tfrecord import gen_tfrecord, load_data

parser = argparse.ArgumentParser()

parser.add_argument('--config', help='config file',
                    default='config.json')
parser.add_argument('--gpu', help='which gpu to use', default='3')
parser.add_argument("--is_train", type=utils.str2bool, default=True, help="is_train or infer&evaluate")

args = parser.parse_args()
poj_base = os.path.dirname(args.config)
logger = utils.get_logger(os.path.join(poj_base, 'main.log'))
config = utils.load_config(args.config)

train_dir = os.path.join(poj_base, config['train_dir'])
data_dir = os.path.join(poj_base, config['data_dir'])
eval_dir = os.path.join(poj_base, config['eval_dir'])
log_dir = os.path.join(poj_base, config['log_dir'])
preprocess_dir = os.path.join(poj_base, config['preprocess_dir'])
best_model = os.path.join(poj_base, config['best_model'])

word_vocab_file = os.path.join(preprocess_dir, config['word_vocab_file'])
intent_vocab_file = os.path.join(preprocess_dir, config['intent_vocab_file'])

train_files = [os.path.join(data_dir, i) for i in config['train_file']]
ind_valid_file = os.path.join(data_dir, config['ind_valid_file'])
ood_valid_file = os.path.join(data_dir, config['ood_valid_file'])
ind_test_file = os.path.join(data_dir, config['ind_test_file'])
ood_test_file = os.path.join(data_dir, config['ood_test_file'])

train_prep_file = [os.path.join(preprocess_dir, os.path.basename(i) + '.pkl') for i in train_files]
ind_valid_prep_file = os.path.join(preprocess_dir, os.path.basename(ind_valid_file) + '.pkl')
ood_valid_prep_file = os.path.join(preprocess_dir, os.path.basename(ood_valid_file) + '.pkl')


def add_summary(writer, step, data):
    for i in data:
        name, value = i
        writer.add_summary(summary=tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)]), global_step=step)


def safe_summ(summ, step, writer):
    if summ is not None:
        writer.add_summary(summ, step)


def log_batch(infer, id2word, logger, intent=None, id2intent=None):
    bs = infer.shape[0]
    res = []
    for i in range(bs):
        utter = infer[i]
        eos_pos = utils._find_eos(utter, utils.EOS_ID)
        if intent is None:
            res.append('{}\t{}'.format(i, ' '.join([id2word[j] for j in utter[:eos_pos]])))
        else:
            res.append('{}\t[{}]\t{}'.format(i, id2intent[intent[i]], ' '.join([id2word[j] for j in utter[:eos_pos]])))
        logger.info(res[-1])
    return res


def eval_cls(sess, model, data):
    start = 0
    res = None
    while start < len(data['len']):
        logits = sess.run(model.cls_infer_logits, feed_dict={
            model.utter: data['utter'][start:start + config['batch_size']],
            model.utter_len: data['len'][start:start + config['batch_size']],
            model.keep_rate: 1.0})
        if res is None:
            res = logits
        else:
            res = np.concatenate((res, logits))
        start += config['batch_size']
    return res


np.random.seed(config['seed'])

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger.info('tf Version: {}'.format(tf.__version__))
    for i in config:
        logger.info('{}: {}'.format(i, config[i]))
    for i in vars(args):
        logger.info('{}: {}'.format(i, getattr(args, i)))

    dirs = [train_dir, eval_dir, log_dir, best_model, preprocess_dir]
    for d in dirs:
        if not os.path.isdir(d):
            logger.info('cannot find {}, mkdiring'.format(d))
            os.makedirs(d)

    if args.is_train:
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            check_list = {'word_vocab': word_vocab_file, 'intent_vocab': intent_vocab_file}
            if tf.train.get_checkpoint_state(train_dir):
                # load trained model
                logger.info('Found check point in {}'.format(train_dir))
                for file in check_list:
                    if not os.path.isfile(check_list[file]):
                        logger.error('Can not find {}'.format(check_list[file]))
                        exit(1)
                logger.info('Loading vocab files {}'.format(check_list))
                word2id, id2word = utils.read_vocab(check_list['word_vocab'], logger, limit=config['word_vocab_size'])
                config['word_vocab_size'] = min(config['word_vocab_size'], len(word2id))
                intent2id, id2intent = utils.read_vocab(check_list['intent_vocab'], logger)
                config = utils.update_vocab_size(config, intent2id, dict())

                logger.info('Loading checkpoint from {}'.format(tf.train.latest_checkpoint(train_dir)))
                model = CGAN(config=config, go_id=utils.GO_ID, eos_id=utils.EOS_ID)
                model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
            else:
                # Begin from scratch
                logger.info('Starting from scratch')
                miss_files = [file for file in check_list.values() if not os.path.isfile(file)]
                if len(miss_files) != 0:
                    logger.info('{} files '.format(len(miss_files)) +
                                'are missing: \n {}'.format('\n'.join(miss_files)))
                    logger.info('re-building vocab')
                    utils.build_vocab(
                        train_files, check_list['word_vocab'], check_list['intent_vocab'], logger, word_level=True)

                logger.info('Loading vocab files')
                word2id, id2word = utils.read_vocab(check_list['word_vocab'], logger, limit=config['word_vocab_size'])
                config['word_vocab_size'] = min(config['word_vocab_size'], len(word2id))
                intent2id, id2intent = utils.read_vocab(check_list['intent_vocab'], logger)
                config = utils.update_vocab_size(config, intent2id, dict())
                embed = utils.load_embed(word2id, config['word_vocab_size'], config['word_embed_size'],
                                         config['pretrained_embed'], logger)
                model = CGAN(config=config, go_id=utils.GO_ID, eos_id=utils.EOS_ID, embed=embed)
                sess.run(tf.global_variables_initializer())

            logger.info('Preparing data')
            if not os.path.isfile(train_prep_file[0]):
                logger.info('{} not found, generating from {}'.format(train_prep_file[0], train_files))
                train_data = load_data(
                    word2id, intent2id, train_files, logger, config['max_utter_len'], word_level=True, shuffle=True)
                with open(train_prep_file[0], 'wb') as f:
                    pickle.dump(train_data, f)
            else:
                logger.info('{} found, loading'.format(train_prep_file[0]))
                with open(train_prep_file[0], 'rb') as f:
                    train_data = pickle.load(f)

            if not os.path.isfile(ind_valid_prep_file):
                logger.info('{} not found, generating from {}'.format(ind_valid_prep_file, ind_valid_file))
                ind_valid_data = load_data(
                    word2id, intent2id, [ind_valid_file], logger, config['max_utter_len'], word_level=True, shuffle=True)
                with open(ind_valid_prep_file, 'wb') as f:
                    pickle.dump(ind_valid_data, f)
            else:
                logger.info('{} found, loading'.format(ind_valid_prep_file))
                with open(ind_valid_prep_file, 'rb') as f:
                    ind_valid_data = pickle.load(f)

            if not os.path.isfile(ood_valid_prep_file):
                logger.info('{} not found, generating from {}'.format(ood_valid_prep_file, ood_valid_file))
                ood_valid_data = load_data(
                    word2id, intent2id, [ood_valid_file], logger, config['max_utter_len'], word_level=True, shuffle=True)
                with open(ood_valid_prep_file, 'wb') as f:
                    pickle.dump(ood_valid_data, f)
            else:
                logger.info('{} found, loading'.format(ood_valid_prep_file))
                with open(ood_valid_prep_file, 'rb') as f:
                    ood_valid_data = pickle.load(f)

            logger.info('train len(utter): {}'.format(train_data['utter'].shape))
            logger.info('ind valid len(utter): {}'.format(ind_valid_data['utter'].shape))
            logger.info('ood valid len(utter): {}'.format(ood_valid_data['utter'].shape))

            train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
            valid_writer = tf.summary.FileWriter(os.path.join(log_dir, 'valid'), sess.graph)

            train_ppl_loss = 0.0
            train_real_cls_acc = 0.0
            train_fake_cls_acc = 0.0
            train_real_D_acc = 0.0
            train_fake_D_acc = 0.0
            train_time = 0.0
            last_improved = 0
            start_time = time.time()

            best_ppl_loss = 1e18
            prev_ppl_loss = [1e18 for _ in range(5)]

            logger.info('Training')

            ae_grad_summ = None
            d_grad_summ = None
            g_grad_summ = None
            cls_grad_summ = None
            regul_g_grad_summ = None
            d_into_ae_grad_summ = None
            ae_scalar_summ = None
            d_scalar_summ = None
            g_scalar_summ = None
            cls_scalar_summ = None

            train_data_size = len(train_data['intent'])
            text_ph = tf.placeholder(dtype=tf.string, shape=[None])
            noise_summ = tf.summary.text('noise', text_ph)
            latent_summ = tf.summary.text('latent', text_ph)
            ae_clean_summ = tf.summary.text('ae_clean', text_ph)
            ae_noise_summ = tf.summary.text('ae_noise', text_ph)
            orig_summ = tf.summary.text('orig', text_ph)

            scalar_ph = tf.placeholder(dtype=tf.float32, shape=[])
            acc_summ = tf.summary.scalar('test_acc', scalar_ph, family='cls_test')
            auroc_summ = tf.summary.scalar('test_auroc', scalar_ph, family='cls_test')
            fpr_tpr95_summ = tf.summary.scalar('test_fpr@tpr95', scalar_ph, family='cls_test')
            auroc_10_summ = tf.summary.scalar('test_auroc_10', scalar_ph, family='cls_test')
            fpr_tpr95_10_summ = tf.summary.scalar('test_fpr@tpr95_10', scalar_ph, family='cls_test')

            for epoch in range(config['max_epoch']):
                start = 0
                while start < train_data_size:
                    start_time = time.time()
                    # train ae and cls
                    ae_scalar_summ, cls_scalar_summ, real_cls_acc, fake_cls_acc, ppl_loss, _, _ = sess.run(
                        [model.ae_scalar_summ_op, model.cls_scalar_summ_op, model.real_cls_acc, model.fake_cls_acc,
                         model.ppl_loss, model.train_ae, model.train_cls],
                        feed_dict={model.utter: train_data['utter'][start: start + config['batch_size']],
                                   model.intents: train_data['intent'][start: start + config['batch_size']],
                                   model.utter_len: train_data['len'][start: start + config['batch_size']],
                                   model.keep_rate: config['keep_rate']})
                    train_ppl_loss += ppl_loss
                    train_real_cls_acc += real_cls_acc
                    train_fake_cls_acc += fake_cls_acc

                    for i in range(config['niter_gan_d']):
                        # train D
                        rand_start = random.randint(0, start)
                        d_scalar_summ, real_D_acc, fake_D_acc, _ = sess.run(
                            [model.d_scalar_summ_op, model.real_D_acc, model.fake_D_acc, model.train_d],
                            feed_dict={model.utter: train_data['utter'][rand_start: rand_start + config['batch_size']],
                                       model.intents: train_data['intent'][rand_start: rand_start + config['batch_size']],
                                       model.utter_len: train_data['len'][rand_start: rand_start + config['batch_size']],
                                       model.keep_rate: config['keep_rate']})
                        train_real_D_acc += real_D_acc
                        train_fake_D_acc += fake_D_acc

                    for i in range(config['niter_gan_g']):
                        # train G
                        rand_start = random.randint(0, start)
                        sess.run(
                            [model.train_g, model.train_regularized_g],
                            feed_dict={model.utter: train_data['utter'][rand_start: rand_start + config['batch_size']],
                                       model.intents: train_data['intent'][rand_start: rand_start + config['batch_size']],
                                       model.utter_len: train_data['len'][rand_start: rand_start + config['batch_size']],
                                       model.keep_rate: config['keep_rate']})

                    # for i in range(config['niter_gan_d_into_ae']):
                        # train D into ae
                        # d_into_ae_grad_summ, _ = sess.run(
                        #     [model.d_into_ae_grad_summ_op, model.train_d_into_ae],
                        #     feed_dict={model.data_handler: train_handler_seen2, model.keep_rate: config['keep_rate']})
                        # _ = sess.run(
                        #     [model.train_d_into_ae],
                        #     feed_dict={model.data_handler: train_handler_seen2, model.keep_rate: config['keep_rate']})

                    ae_step = sess.run(model.ae_step)
                    if ae_step % config['summary_per_step'] == 0:
                        safe_summ(ae_grad_summ, ae_step, train_writer)
                        safe_summ(d_grad_summ, ae_step, train_writer)
                        safe_summ(g_grad_summ, ae_step, train_writer)
                        safe_summ(cls_grad_summ, ae_step, train_writer)
                        safe_summ(regul_g_grad_summ, ae_step, train_writer)
                        safe_summ(d_into_ae_grad_summ, ae_step, train_writer)
                        safe_summ(ae_scalar_summ, ae_step, train_writer)
                        safe_summ(d_scalar_summ, ae_step, train_writer)
                        safe_summ(g_scalar_summ, ae_step, train_writer)
                        safe_summ(cls_scalar_summ, ae_step, train_writer)

                    train_time += (time.time() - start_time)

                    if ae_step % config['save_per_step'] == 0:
                        train_ppl_loss /= (config['save_per_step'] * config['niter_ae'])
                        train_real_cls_acc /= (config['save_per_step'] * config['niter_ae'])
                        train_fake_cls_acc /= (config['save_per_step'] * config['niter_ae'])
                        if config['niter_gan_d'] != 0:
                            train_real_D_acc /= (config['save_per_step'] * config['niter_gan_d'])
                            train_fake_D_acc /= (config['save_per_step'] * config['niter_gan_d'])
                        train_time /= config['save_per_step']
                        model.saver.save(sess, os.path.join(train_dir, 'model.ckpt'), global_step=ae_step)
                        format_str = 'It: {0:>5} t ppl: {1:>4.4f} t real cls_acc: {2:3.4f} t fake cls acc: {3:3.4f} t real_D_acc: {4:>3.4f} ' + \
                                     't_fake_D_acc {5:3.4f} time {6:3.4f}'
                        out1 = format_str.format(ae_step, np.exp(train_ppl_loss), train_real_cls_acc, train_fake_cls_acc, train_real_D_acc,
                                                 train_fake_D_acc, train_time * 1000)

                        ind_logits = eval_cls(sess, model, ind_valid_data)
                        ood_logits = eval_cls(sess, model, ood_valid_data)
                        ind_sm = eval.t_scaling_softmax(ind_logits, 1)
                        ood_sm = eval.t_scaling_softmax(ood_logits, 1)
                        ind_sm_10 = eval.t_scaling_softmax(ind_logits, 10)
                        ood_sm_10 = eval.t_scaling_softmax(ood_logits, 10)
                        acc = sklearn.metrics.accuracy_score(ind_valid_data['intent'], np.argmax(ind_logits, axis=1))
                        auroc = eval.cal_roc_auc(ind_sm, ood_sm)
                        fpr_at_tpr95 = eval.fpr_at_tpr95(ind_sm, ood_sm)[0]
                        auroc_10 = eval.cal_roc_auc(ind_sm_10, ood_sm_10)
                        fpr_at_tpr95_10 = eval.fpr_at_tpr95(ind_sm_10, ood_sm_10)[0]

                        format_str = 'v cls_acc: {0:3.4f} v auroc: {1:3.4f} v fpr@tpr95: {2:3.4f} v10 auroc: {3:3.4f} v10 fpr@tpr95: {4:3.4f}'

                        out2 = format_str.format(acc, auroc, fpr_at_tpr95, auroc_10, fpr_at_tpr95_10)
                        logger.info(out1)
                        logger.info(out2)

                        train_writer.add_summary(sess.run(acc_summ, feed_dict={scalar_ph: acc}), ae_step)
                        train_writer.add_summary(sess.run(auroc_summ, feed_dict={scalar_ph: auroc}), ae_step)
                        train_writer.add_summary(sess.run(fpr_tpr95_summ, feed_dict={scalar_ph: fpr_at_tpr95}), ae_step)
                        train_writer.add_summary(sess.run(auroc_10_summ, feed_dict={scalar_ph: auroc_10}), ae_step)
                        train_writer.add_summary(sess.run(fpr_tpr95_10_summ, feed_dict={scalar_ph: fpr_at_tpr95_10}), ae_step)

                        logger.info("---------noise sample--------------")
                        noise_sample = sess.run(model.noise_greedy_utter,
                                                feed_dict={model.batch_size: 5, model.keep_rate: 1.0})
                        noise_res = log_batch(noise_sample, id2word, logger, intent=None, id2intent=id2intent)
                        train_writer.add_summary(sess.run(noise_summ, feed_dict={text_ph: noise_res}), ae_step)
                        logger.info("---------latent sample--------------")
                        greedy_sample = sess.run(model.latent_greedy_utter,
                                                 feed_dict={model.batch_size: 5, model.keep_rate: 1.0})
                        latent_res = log_batch(greedy_sample, id2word, logger, intent=None, id2intent=id2intent)
                        train_writer.add_summary(sess.run(latent_summ, feed_dict={text_ph: latent_res}), ae_step)
                        logger.info("----------noise ae sample--------------")

                        rand_start = random.randint(0, start)
                        orig_utter, clear_sample, noise_sample, intent = sess.run(
                            [model.utter, model.infer_clear_greedy_utter, model.infer_noise_greedy_utter, model.intents],
                            feed_dict={model.utter: train_data['utter'][rand_start: rand_start + 5],
                                       model.intents: train_data['intent'][rand_start: rand_start + 5],
                                       model.utter_len: train_data['len'][rand_start: rand_start + 5],
                                       model.keep_rate: 1.0})
                        ae_noise_res = log_batch(noise_sample, id2word, logger, intent, id2intent)
                        train_writer.add_summary(sess.run(ae_noise_summ, feed_dict={text_ph: ae_noise_res}), ae_step)
                        logger.info("----------clear ae sample--------------")
                        ae_clean_res = log_batch(clear_sample, id2word, logger, intent, id2intent)
                        train_writer.add_summary(sess.run(ae_clean_summ, feed_dict={text_ph: ae_clean_res}), ae_step)
                        logger.info('--------original utter----------')
                        orig_res = log_batch(orig_utter, id2word, logger, intent, id2intent)
                        train_writer.add_summary(sess.run(orig_summ, feed_dict={text_ph: orig_res}), ae_step)

                        print(out1)
                        print(out2)

                        train_ppl_loss = 0.0
                        train_real_cls_acc = 0.0
                        train_fake_cls_acc = 0.0
                        train_real_D_acc = 0.0
                        train_fake_D_acc = 0.0
                        train_time = 0.0

                    start = start + config['batch_size']

    else:
        ckpt_dir = train_dir
        ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
        if not ckpt_file:
            logger.error("cannot find checkpoint in {}".format(ckpt_dir))
            exit(1)
        logger.info("found checkpoint {}".format(ckpt_file))

        word2id, id2word = utils.read_vocab(word_vocab_file, logger, config['word_vocab_size'])
        intent2id, id2intent = utils.read_vocab(intent_vocab_file, logger)
        config = utils.update_vocab_size(config, intent2id, dict())

        logger.info('loading ind test data from {}'.format(ind_test_file))
        ind_data = load_data(word2id, intent2id, [ind_test_file], logger, config['max_utter_len'], word_level=True, shuffle=True)
        logger.info('loading ind test data from {}'.format(ood_test_file))
        ood_data = load_data(word2id, intent2id, [ood_test_file], logger, config['max_utter_len'], word_level=True, shuffle=True)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf.reset_default_graph()
        with tf.Session(config=tf_config) as sess:
            logger.info('loading model from {}'.format(ckpt_file))
            model = CGAN(config=config, go_id=utils.GO_ID, eos_id=utils.EOS_ID)
            model.saver.restore(sess, ckpt_file)

            ind_logits = None
            start = 0
            while start < len(ind_data['len']):
                logits = sess.run(model.cls_infer_logits, feed_dict={
                    model.utter: ind_data['utter'][start:start + config['batch_size']],
                    model.utter_len: ind_data['len'][start:start + config['batch_size']],
                    model.keep_rate: 1.0})
                if ind_logits is None:
                    ind_logits = logits
                else:
                    ind_logits = np.concatenate((ind_logits, logits))
                start += config['batch_size']

            start = 0
            ood_logits = eval_cls(sess, model, ood_data)

            ind_sm = eval.t_scaling_softmax(ind_logits, 1)
            ood_sm = eval.t_scaling_softmax(ood_logits, 1)
            ind_sm_10 = eval.t_scaling_softmax(ind_logits, 10)
            ood_sm_10 = eval.t_scaling_softmax(ood_logits, 10)
            logger.info('Ind acc: {}'.format(sklearn.metrics.accuracy_score(
                ind_data['intent'], np.argmax(ind_sm, axis=1))))
            logger.info('AUROC: {}'.format(eval.cal_roc_auc(ind_sm, ood_sm)))
            logger.info('fpr@tpr95: {}'.format(eval.fpr_at_tpr95(ind_sm, ood_sm)))
            logger.info('AUROC t_10: {}'.format(eval.cal_roc_auc(ind_sm_10, ood_sm_10)))
            logger.info('fpr@tpr95 t_10: {}'.format(eval.fpr_at_tpr95(ind_sm_10, ood_sm_10)))

except:
    logger.error(traceback.format_exc())




