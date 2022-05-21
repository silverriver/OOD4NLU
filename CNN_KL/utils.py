import os
import logging
import collections
import numpy as np
import random
import json
import argparse
import tensorflow as tf

_PAD = "<_PAD>"
_GO = "<_GO>"
_EOS = "<_EOS>"
_UNK = "<_UNK>"
_OOD = "<_OOD>"
_START_VOCAB = [_PAD, _EOS, _GO, _UNK, _OOD]

PAD_ID = 0
EOS_ID = 1
GO_ID = 2
UNK_ID = 3
OOD_ID = 4


def mean_var(l):
    l = np.asarray(l)
    mean = np.sum(l) / len(l)
    var = np.sqrt(np.sum((l - mean) ** 2) / len(l))
    return mean, var


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_logger(filename, print2screen=True):
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s][%(thread)d][%(filename)s][line: %(lineno)d][%(levelname)s] \
>> %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    if print2screen:
        logger.addHandler(ch)
    return logger


def build_vocab(files, char_vocab_file, intent_vocab_file, logger, word_level=True):
    for train_file in files:
        if not os.path.isfile(train_file):
            logger.error("can not find {}".format(train_file))
            return
    char_vocab = collections.Counter()
    intent_vocab = collections.Counter()
    count = 0
    for train_file in files:
        logger.info('reading {}'.format(train_file))
        with open(train_file, encoding='utf-8') as file:
            for i in file:
                i = i.strip()
                if len(i) == 0:
                    continue
                count += 1
                i = i.split('\t')

                intent_vocab.update([i[0].lower()])

                if word_level:
                    char_vocab.update(i[1].split())
                else:
                    char_vocab.update(i[1].replace(' ', ''))

    logger.info('{} utts handled'.format(count))
    logger.info('{} char found'.format(len(char_vocab)))
    logger.info('{} intent found'.format(len(intent_vocab)))

    if not os.path.isfile(char_vocab_file):
        with open(char_vocab_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(_START_VOCAB + [i[0] for i in char_vocab.most_common()]))
            f.write('\n')

    with open(intent_vocab_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join([i[0] for i in intent_vocab.most_common()]))
        f.write('\n')

    logger.info('write to file \n{}, \n{}'.format(char_vocab_file, intent_vocab_file))


def load_embed(word2index, char_vocab_size, embed_size, pretrained_embed_file, logger):
    if not os.path.exists(pretrained_embed_file):
        logger.info("Cannot find word vector file")
        return None

    logger.info("Loading word vectors...")
    embed = [np.zeros(embed_size, dtype=np.float32) for _ in range(char_vocab_size)]
    count = 0
    with open(pretrained_embed_file) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                logger.info("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ') + 1:]
            if word in word2index:
                embed[word2index[word]] = list(map(float, vector.split()))
                count += 1
    embed = np.array(embed, dtype=np.float32)
    logger.info("{} word vectors ({}) pre-trained".format(count, count / len(word2index)))
    return embed


def read_vocab(vocab_file, logger, limit=-1):
    if not os.path.isfile(vocab_file):
        logger.error('cannot find {}'.format(vocab_file))
        return None, None

    with open(vocab_file, encoding='utf-8') as f:
        vocab = [i.strip() for i in f.readlines() if len(i.strip()) != 0]

    if limit != -1:
        vocab = vocab[:limit]

    logger.info('{} loaded, vocab size {}'.format(vocab_file, len(vocab)))
    return dict(zip(vocab, range(len(vocab)))), dict(zip(range(len(vocab)), vocab))


def read_char_freq(char_freq_file, logger, vocab):
    if not os.path.isfile(char_freq_file):
        logger.error('cannot find {}'.format(char_freq_file))
        return None, None

    with open(char_freq_file, encoding='utf-8') as f:
        res = [i.strip().split() for i in f.readlines() if len(i.strip()) != 0]
        res = [i for i in res if len(i) == 2]

    char_list, char_probs = [], []
    for i in res:
        if i[0] in vocab:
            char_list.append(vocab[i[0]])
            char_probs.append(int(i[1]))

    count = sum(char_probs)
    char_probs = [i / count for i in char_probs]
    logger.info('{} loaded, vocab size {}'.format(char_freq_file, len(char_list)))
    return char_list, char_probs


def update_vocab_size(config, intent2id):
    config['intent_num'] = len(intent2id)
    return config


def _eval_ind(sess, model, data_handler):
    valid_intent_loss = 0.0
    valid_acc = 0.0
    res = dict()
    res['intent_gt'] = []
    res['intent_pred'] = []
    res['intent_softmax'] = None
    res['intent_logits'] = None
    res['cross_entropy'] = None
    res['bs'] = 0
    res['ids'] = []

    while True:
        try:
            input_feed = {model.ind_data_handler: data_handler, model.keep_rate: 1.0}
            output_feed = [model.intent, model.ind_bs, model.acc, model.intent_predict, model.intent_softmax,
                           model.intent_logits_ind, model.cross_entropy, model.intent_loss, model.ind_ids]

            intent_gt, ind_bs, acc, pred_label, intent_softmax, intent_logits, cross_entropy, \
                intent_loss, ids = sess.run(output_feed, input_feed)

            res['bs'] += ind_bs
            res['intent_gt'] = np.concatenate((res['intent_gt'], intent_gt))
            res['intent_pred'] = np.concatenate((res['intent_pred'], pred_label))
            res['ids'] = np.concatenate((res['ids'], ids))
            if res['cross_entropy'] is None:
                res['cross_entropy'] = cross_entropy
            else:
                res['cross_entropy'] = np.concatenate((res['cross_entropy'], cross_entropy))

            if res['intent_softmax'] is None:
                res['intent_softmax'] = intent_softmax
            else:
                res['intent_softmax'] = np.concatenate((res['intent_softmax'], intent_softmax))

            if res['intent_logits'] is None:
                res['intent_logits'] = intent_logits
            else:
                res['intent_logits'] = np.concatenate((res['intent_logits'], intent_logits))

            valid_intent_loss += intent_loss * ind_bs
            valid_acc += acc * ind_bs
        except tf.errors.OutOfRangeError:
            break

    valid_intent_loss /= res['bs']
    valid_acc /= res['bs']
    return valid_intent_loss, valid_acc, res


def _eval_ood(sess, model, data_handler):
    valid_kl_loss = 0.0
    res = dict()
    res['intent_logits_ood'] = None
    res['kl_term'] = None
    res['ids'] = []
    res['bs'] = 0

    while True:
        try:
            input_feed = {model.ood_data_handler: data_handler, model.keep_rate: 1.0}
            output_feed = [model.intent_logits_ood, model.kl_term, model.kl_loss, model.ood_bs, model.ood_ids]
            intent_logits_ood, kl_term, kl_loss, ood_bs, ids = sess.run(output_feed, input_feed)

            res['bs'] += ood_bs
            res['ids'] = np.concatenate((res['ids'], ids))

            if res['intent_logits_ood'] is None:
                res['intent_logits_ood'] = intent_logits_ood
            else:
                res['intent_logits_ood'] = np.concatenate((res['intent_logits_ood'], intent_logits_ood), axis=0)

            if res['kl_term'] is None:
                res['kl_term'] = kl_term
            else:
                res['kl_term'] = np.concatenate((res['kl_term'], kl_term), axis=0)

            valid_kl_loss += kl_loss * ood_bs
        except tf.errors.OutOfRangeError:
            break

    valid_kl_loss /= res['bs']
    return valid_kl_loss, res


def _get_utt(sess, model, data_handler):
    res = dict()
    res['bs'] = 0
    res['utt'] = None
    while True:
        try:
            bs, utt = sess.run([model.ind_utt, model.ind_bs], {model.ind_data_handler: data_handler})
            res['bs'] += bs
            if res['utt'] is None:
                res['utt'] = utt
            else:
                max_len = max(res['utt'].shape[1], utt.shape[1])

                def _pad(input):
                    if max_len <= input.shape[1]:
                        return input
                    return np.concatenate((input, np.ones((input.shape[0], max_len - input.shape[1]),
                                                          dtype=np.int32) * PAD_ID), axis=1)
                res['utt'] = np.concatenate((_pad(res['utt']), _pad(utt)), axis=0)
        except tf.errors.OutOfRangeError:
            break
    return res


def evaluate_file(sess, model, ind_files, ood_files, ind_dataset, ood_dataset, bs, save_utt=False):
    ind_dataset.init(sess, [ind_files], bs)
    ood_dataset.init(sess, [ood_files], bs)
    ind_handler = ind_dataset.get_handler(sess)
    ood_handler = ood_dataset.get_handler(sess)

    ind_intent_loss, ind_acc, ind_res = _eval_ind(sess, model, ind_handler)
    ind_dataset.init(sess, [ind_files], bs)
    ind_kl_loss, ind_res_2 = _eval_ood(sess, model, ind_handler)
    assert ind_res['bs'] == ind_res_2['bs']
    ind_res.update(ind_res_2)

    valid_loss, valid_acc, valid_intent_loss, valid_slot_loss, valid_slot_acc, \
        valid_ind_kl_loss, ind_res = _eval_ind(sess, model, ind_iter, save_utt=save_utt)
    valid_ood_kl_loss, ood_res = _eval_ood(sess, model, ood_iter, save_utt=save_utt)

    return valid_acc, valid_loss, valid_slot_acc, valid_intent_loss, valid_slot_loss, \
               valid_ind_kl_loss, valid_ood_kl_loss, ind_res, ood_res


def load_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
        return config


def show_utt(id2word, utt):
    return ''.join([id2word[i] for i in utt])


if __name__ == '__main__':
    import sys
    iter = DataIterRand(dict([[1,2], [2,3], [_EOS, 6], [_PAD, 7]]), 5, 1, 5, bs=3)
    d = iter.next()


    infile = 'data/wechat_root_ind_dev'
    char_vocab = os.path.join(os.path.dirname(infile), 'char_vocab')
    state_vocab = os.path.join(os.path.dirname(infile), 'state_vocab')
    intent_vocab = os.path.join(os.path.dirname(infile), 'intent_vocab')
    slot_vocab = os.path.join(os.path.dirname(infile), 'slot_vocab')

    log = get_logger('test.log')
    build_vocab([infile],
                char_vocab_file=char_vocab,
                state_vocab_file=state_vocab,
                intent_vocab_file=intent_vocab,
                slot_vocab_file=slot_vocab,
                logger=log)
    char2id, id2char = read_vocab(char_vocab, log)
    print(char2id)
    print(id2char)
    intent2id, id2intent = read_vocab(intent_vocab, log)
    print(intent2id)
    print(id2intent)
    slot2id, id2slot = read_vocab(slot_vocab, log)
    print(slot2id)
    print(id2slot)
    state2id, id2state = read_vocab(state_vocab, log)
    print(state2id)
    print(id2state)

    data = DataIter(infile, char2id, intent2id, slot2id, state2id, 50, 3000)
    while data.epoch < 4:
        print('data.epoch', data.epoch)
        d = data.next()
        print(d['bs'], len(d['intent']))
    data.reset()

    print('------------')
    while data.epoch < 4:
        print('data.epoch', data.epoch)
        d = data.next()
        print(d['bs'], len(d['intent']))




