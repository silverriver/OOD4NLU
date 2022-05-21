import tensorflow as tf
import utils
import numpy as np
import collections
import os
import sys
import random


class Tokenizer:
    def __init__(self, word_level=True):
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


def _safe_list2ids(vocab, unk_id, tokens):
    return [vocab[t] if t in vocab else unk_id for t in tokens]


# ref: https://stackoverflow.com/questions/47939537/how-to-use-dataset-api-to-read-tfrecords-file-of-lists-of-variant-length
# ref: https://blog.csdn.net/qq1483661204/article/details/78932389
def gen_tfrecord(vocab, intent2id, infiles, outfiles, logger, max_len, word_level=True, shuffle=False):
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
            inputs = [i.strip().split('\t') for i in f.readlines() if len(i.strip()) != 0]
            inputs = [i for i in inputs if len(i) == 2]
            if shuffle:
                random.shuffle(inputs)

            for input in inputs:
                utt = tokenizer.tokenize(input[1])
                utt = utt[:max_len - 1] + [utils._EOS]
                if len(utt) == 0:
                    continue
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
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    infiles =  ["top100_email_root_ind_dev",
                "top100_email_root_ind_test"]
    outfiles = ["top100_email_root_ind_dev.tfrecord",
                "top100_email_root_ind_test.tfrecord"]

    char_vocab_file =   "char_vocab"
    intent_vocab_file = "intent_vocab"
    logger = utils.get_logger("test")

    utils.build_vocab(infiles, char_vocab_file, '/tmp/slot', intent_vocab_file, logger)

    char2ids, ids2char = utils.read_vocab(char_vocab_file, logger)
    intent2ids, ids2intent = utils.read_vocab(intent_vocab_file, logger)

# def gen_tfrecord(vocab, intent2id, infiles, outfiles, logger, max_len, word_level=False):

    gen_tfrecord(char2ids, intent2ids, infiles, outfiles, logger, 100, word_level=True)

    check_tfrecord(ids2char, ids2intent, outfiles[0], 3)

    exit(0)
    def parse(example):
        name_to_features = {
            "utter": tf.VarLenFeature(tf.int64),
            "intent": tf.FixedLenFeature([], tf.int64),
            "len": tf.FixedLenFeature([], tf.int64),
        }
        data = tf.parse_single_example(example, name_to_features)
        for name in data.keys():
            if name == "utter":
                t = tf.sparse_tensor_to_dense(data[name])
            else:
                t = data[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            data[name] = tf.reshape(t, [-1])
        return data

    def convert_reshape(data):
        for name in data.keys():
            t = data[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            data[name] = tf.reshape(t, [-1])
        return data


    padded_shape = {
        "utter": [None],
        "intent": [None],
        "len": [None],
    }

    padded_value = {
        "utter": tf.constant(char2ids[utils._PAD], tf.int32),
        "intent": tf.constant(-1, tf.int32),
        "len": tf.constant(-1, tf.int32),
    }

    bs = tf.placeholder(tf.int64, shape=[])

    def _construct_dataset(data_set):
        data_set = data_set.map(parse, num_parallel_calls=1)
        data_set = data_set.padded_batch(bs, padded_shapes=padded_shape, padding_values=padded_value)
        return data_set

    file_ph = tf.placeholder(tf.string, shape=[None])
    data_set = tf.data.TFRecordDataset(file_ph)
    data_set = _construct_dataset(data_set)
    data_iter = data_set.make_initializable_iterator()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(data_iter.initializer, feed_dict={file_ph: outfiles, bs: 3})

    dataset_handle = sess.run(data_iter.string_handle())

    handle = tf.placeholder(tf.string, shape=[])
    iter = tf.data.Iterator.from_string_handle(handle, data_set.output_types)

    np_utter = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [2,2,34,1], [23, 34,21,123]])
    np_len = np.array([4, 4, 4, 4])
    np_intent = np.array([6, 7, 2, 6])
    # np_len = np.array([[4], [4], [4], [4]])
    # np_intent = np.array([[6], [7], [2], [6]])

    np_utter_ph = tf.placeholder(tf.int32, [None, None])
    np_len_ph = tf.placeholder(tf.int32, [None])
    np_intent_ph = tf.placeholder(tf.int32, [None])
    np_dataset = tf.data.Dataset.from_tensor_slices({"utter": np_utter_ph,
                                                     "len": np_len_ph,
                                                     "intent": np_intent_ph})

    np_dataset = np_dataset.map(convert_reshape)
    np_dataset = np_dataset.batch(bs)
    # np_dataset = np_dataset.padded_batch(bs, padded_shapes=padded_shape, padding_values=padded_value)
    np_dataiter = np_dataset.make_initializable_iterator()
    sess.run(np_dataiter.initializer, feed_dict={bs: 2,
                                                 np_utter_ph: np_utter,
                                                 np_len_ph: np_len,
                                                 np_intent_ph: np_intent})

    np_data_handle = sess.run(np_dataiter.string_handle())

    class Model:
        def __init__(self, iter):
            self.input = iter.get_next()
            self.output = tf.reduce_sum(self.input['utter'])

    model = Model(iter)

    for i in range(5):
        print('from file', i)
        dataset_handle = sess.run(data_iter.string_handle())
        res = sess.run(model.output, feed_dict={handle: dataset_handle})
        print(res)

    for i in range(2):
        print('from np', i)
        res = sess.run(model.output, feed_dict={handle: np_data_handle})
        print(res)


