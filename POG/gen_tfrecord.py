import tensorflow as tf
import utils
import numpy as np
import os
import random


class Tokenizer:
    def __init__(self, word_level=False):
        self.word_level = word_level

    def tokenize(self, utter):
        if self.word_level:
            return utter.split()
        else:
            return list(''.join(utter.split()))


def _safe_token2id(vocab, unk_id, token):
    if token in vocab:
        return vocab[token]
    else:
        return unk_id


def _pad_utter(utter, pad_id, max_len):
    return utter + [pad_id] * (max_len - len(utter))


def _safe_list2ids(vocab, unk_id, tokens):
    return [vocab[t] if t in vocab else unk_id for t in tokens]


# ref: https://stackoverflow.com/questions/47939537/how-to-use-dataset-api-to-read-tfrecords-file-of-lists-of-variant-length
# ref: https://blog.csdn.net/qq1483661204/article/details/78932389
def gen_tfrecord(vocab, intent2id, infiles, outfiles, logger, max_len, word_level=False, shuffle=False):
    """
    generate tfrecord files based on the file given by infile.
    Only the index of each word is saved
    :param vocab: dict from word to index
    :param infiles: input file
    :param outfiles: output file
    :param logger: logger to tell the world what happened
    :param word_level: ture to use word level input
    :return:
    Returns nothing, but generate a outfile file for the generated tfrecord
    """
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    tokenizer = Tokenizer(word_level=word_level)
    assert len(infiles) == len(outfiles)

    writers = []
    for outfile in outfiles:
        writers.append(tf.python_io.TFRecordWriter(outfile))

    count = 0
    for index, infile in enumerate(infiles):
        logger.info('handling {}:{}'.format(index, infile))
        with open(infile) as f:
            inputs = [i.split('\t') for i in f.readlines() if len(i.strip()) != 0]
            inputs = [i for i in inputs if len(i) == 2]
            if shuffle:
                random.shuffle(inputs)

            for input in inputs:
                utt = tokenizer.tokenize(input[1])
                if len(utt) == 0:
                    continue
                utt = [utils._GO] + utt[:max_len - 2] + [utils._EOS]
                utt = _safe_list2ids(vocab, vocab[utils._UNK], utt)
                intent = _safe_token2id(intent2id, -1, input[0].lower())

                features = dict()
                features["utter"] = _int64_feature(utt)
                features["len"] = _int64_feature([len(utt)])
                features["intent"] = _int64_feature([intent])
                features["id"] = _int64_feature([count])
                count += 1

                example = tf.train.Example(features=tf.train.Features(feature=features))
                writers[index].write(example.SerializeToString())

    for index, writer in enumerate(writers):
        logger.info('Finished, generated tfrecord_file: {}'.format(outfiles[index]))
        writer.close()


def load_data(vocab, intent2id, infiles, logger, max_len, word_level=False, shuffle=False):
    tokenizer = Tokenizer(word_level=word_level)

    count = 0
    data = []

    for index, infile in enumerate(infiles):
        logger.info('handling {}:{}'.format(index, infile))
        with open(infile) as f:
            inputs = [i.split('\t') for i in f.readlines() if len(i.strip()) != 0]
            inputs = [i for i in inputs if len(i) == 2]

            for input in inputs:
                utt = tokenizer.tokenize(input[1])
                if len(utt) == 0:
                    continue
                utt = [utils._GO] + utt[:max_len - 2] + [utils._EOS]
                utt = _safe_list2ids(vocab, vocab[utils._UNK], utt)
                res = [len(utt)]
                utt = _pad_utter(utt, vocab[utils._PAD], max_len)
                intent = _safe_token2id(intent2id, -1, input[0].lower())
                res.append(utt)
                res.append(intent)
                res.append(count)
                data.append(res)
                count += 1

    if shuffle:
        random.shuffle(data)

    return {"utter": np.asarray([i[1] for i in data], dtype=np.int32),
            "len": np.asarray([i[0] for i in data], dtype=np.int32),
            "intent": np.asarray([i[2] for i in data], dtype=np.int32),
            "id": np.asarray([i[3] for i in data], dtype=np.int32)}


def check_tfrecord(ids2char, ids2intent, file, n):
    record_iter = tf.python_io.tf_record_iterator(path=file)
    count = 0
    for record in record_iter:
        if count > n:
            break
        example = tf.train.Example()
        example.ParseFromString(record)
        print("utter", [ids2char[i] for i in example.features.feature["utter"].int64_list.value])
        print("len", example.features.feature["len"].int64_list.value)
        print("intent", [ids2intent[i] for i in example.features.feature["intent"].int64_list.value])
        count += 1


if __name__ == '__main__':
    logger = utils.get_logger("test")
    infile = ["data/dl_root_ind_dev"]
    vocab_file = "word_vocab"
    intent_vocab_file = "intent_vocab"
    word2id, id2word = utils.read_vocab(vocab_file, logger)
    intent2id, id2intent = utils.read_vocab(intent_vocab_file, logger)
    data = load_data(word2id, intent2id, infile, logger, 100, word_level=True, shuffle=True)

    print(data.keys())
    for i in data:
        print(i, data[i].shape)
