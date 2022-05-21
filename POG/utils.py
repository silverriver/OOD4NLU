import os
import logging
import collections
import numpy as np
import random
import json
import argparse
import pickle
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


def count_word(files, outfile, logger, word_level=True):
    for train_file in files:
        if not os.path.isfile(train_file):
            logger.error("can not find {}".format(train_file))
            return
    vocab = collections.Counter()
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
                if len(i) < 4:
                    continue

                if word_level:
                    vocab.update(i[2].split())
                else:
                    vocab.update(i[2].replace(' ', ''))

    logger.info('{} utts handled'.format(count))
    logger.info('{} word found'.format(len(vocab)))
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write('\n'.join([str(i[0]) + '\t' + str(i[1]) for i in vocab.most_common()]))
        f.write('\n')

    logger.info('write to file \n{}'.format(outfile))


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

                intent_vocab.update([i[0]])

                if word_level:
                    char_vocab.update(i[1].split())
                else:
                    char_vocab.update(i[1].replace(' ', ''))

    logger.info('{} utts handled'.format(count))
    logger.info('{} char found'.format(len(char_vocab)))
    logger.info('{} intent found'.format(len(intent_vocab)))
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


def build_preprocess(files, rates, outfile, char2id, slot2id, intent2id, logger):
    logger.info('Building preprocessed files: {}'.format(outfile))

    for infile in files:
        if not os.path.isfile(infile):
            logger.error('cannot find {}'.format(infile))
            return

    data = []
    for infile, rate in zip(files, rates):
        logger.info('handling: {} with rate: {}'.format(infile, rate))
        with open(infile, encoding='utf-8') as f:
            res = [i.strip().lower() for i in f.readlines() if len(i.strip()) != 0]
            for text in res[:int(len(res) * rate)]:
                text = text.lower().split('\t')
                if len(text) < 4:
                    continue
                if 'other' in text[1].lower():
                    text[1] = 'other'
                df = dict()
                if text[1] in intent2id:
                    df['intent'] = intent2id[text[1]]
                else:
                    df['intent'] = -1
                df['utt'] = [char2id[i] if i in char2id else char2id[_UNK] for i in text[2].replace(' ', '')]
                df['utt'] += [char2id[_EOS]]
                slot = set([i.split('b-')[1] for i in text[3].split() if 'b-' in i])
                slot = set([slot2id[i] for i in slot if i in slot2id])
                df['slot'] = [1 if i in slot else 0 for i in range(max(slot2id.values()) + 1)]
                data.append(df)
    with open(outfile, 'wb') as f:
        pickle.dump(data, f)


def update_vocab_size(config, intent2id, slot2id):
    config['slot_num'] = len(slot2id)
    config['intent_num'] = len(intent2id)
    return config


def evaluate(sess, model, valid_handler):
    step_count = 0
    loss = 0
    while True:
        try:
            ppl_loss = sess.run(model.ppl_loss, feed_dict={model.keep_rate: 1.0, model.data_handler: valid_handler})
            step_count += 1
            loss += ppl_loss
        except tf.errors.OutOfRangeError:
            break
    return loss / step_count


def load_config(config_file):
    with open(config_file) as f:
        config = json.load(f)
        return config


def show_utt(id2word, utt):
    return ''.join([id2word[i] for i in utt])


def _find_eos(u, eos_id):
    for i, t in enumerate(u):
        if t == eos_id:
            return i
    return -1


def _half_len(example):
    # example['len'] = [2]
    example['len'] = tf.cast(example['len'] / 2, tf.int32)
    return example


def peep_output(intents, id2intent, id2word, utter, greedy_infer, beam_infer, logger):
    beam_width = beam_infer.shape[-1]
    for i in range(len(intents)):
        logger.info('-----{}: {}------'.format(i, id2intent[intents[i]]))
        eos_pos = _find_eos(utter[i], EOS_ID)
        logger.info('orig   utter: {}'.format(' '.join([id2word[j] for j in utter[i][1:eos_pos]])))
        eos_pos = _find_eos(greedy_infer[i], EOS_ID)
        logger.info('greedy utter: {}'.format(' '.join([id2word[j] for j in greedy_infer[i][:eos_pos]])))
        logger.info('beam   utter:')
        for j in range(beam_width):
            eos_pos = _find_eos(beam_infer[i, :, j], EOS_ID)
            logger.info('{}: {}'.format(j, ' '.join([id2word[k] for k in beam_infer[i, :eos_pos, j]])))



if __name__ == '__main__':
    import sys
    iter = DataIterRand(dict([[1,2], [2,3], [_EOS, 6], [_PAD, 7]]), 5, 1, 5, bs=3)
    d = iter.next()


    infile = 'wechat_root_ind_dev'
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




