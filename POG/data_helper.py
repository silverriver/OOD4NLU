import tensorflow as tf
import numpy as np
# sys.path.append('../..')


def _parse_tfrecord(example):
    name_to_features = {
        "utter": tf.VarLenFeature(tf.int64),
        "intent": tf.FixedLenFeature([], tf.int64),
        "len": tf.FixedLenFeature([], tf.int64),
        "id": tf.FixedLenFeature([], tf.int64),
    }
    data = tf.parse_single_example(example, name_to_features)
    for name in data.keys():
        if name == "utter":
            t = tf.sparse.to_dense(data[name])
            # t = tf.sparse_tensor_to_dense(data[name])
        else:
            t = data[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        data[name] = tf.reshape(t, [-1])
    print(tf.get_default_graph())
    return data


def _suqeeze(example):
    example['len'] = tf.squeeze(example['len'])
    example['id'] = tf.squeeze(example['id'])
    example['intent'] = tf.squeeze(example['intent'])
    return example


class TFRData:
    def __init__(self, PAD_ID, shuffle_buffer=-1, num_parallel_calls=32, repeat=False, prefetch=200,
                 seed=530, convert_fun=None):
        padded_shape = {"utter": [None], "intent": [None], "len": [None], "id": [None]}
        padding_value = {"utter": tf.constant(PAD_ID, tf.int32), "intent": tf.constant(-1, tf.int32),
                         "len": tf.constant(-1, tf.int32), "id": tf.constant(-1, tf.int32)}

        self.file_ph = tf.placeholder(tf.string, shape=[None])
        self.bs = tf.placeholder(tf.int64, shape=[])
        self.seed = seed
        self.data_set = tf.data.TFRecordDataset(self.file_ph)
        self.data_set = self.data_set.map(_parse_tfrecord, num_parallel_calls=num_parallel_calls)
        if convert_fun:
            self.data_set = self.data_set.map(convert_fun, num_parallel_calls=num_parallel_calls)

        if shuffle_buffer > 0:
            self.data_set = self.data_set.shuffle(buffer_size=shuffle_buffer, seed=self.seed)

        self.data_set = self.data_set.padded_batch(self.bs, padded_shapes=padded_shape, padding_values=padding_value)
        self.data_set = self.data_set.map(_suqeeze, num_parallel_calls=num_parallel_calls)

        if repeat is True:
            self.data_set = self.data_set.repeat()
        elif type(repeat) is int:
            self.data_set = self.data_set.repeat(repeat)

        if prefetch > 0:
            self.data_set = self.data_set.prefetch(buffer_size=prefetch)

        self.data_iter = self.data_set.make_initializable_iterator()
        self.is_init = False

    def init(self, sess, infiles, batch_size):
        self.is_init = True
        sess.run(self.data_iter.initializer, feed_dict={self.file_ph: infiles, self.bs: batch_size})

    def get_handler(self, sess):
        assert self.init
        return sess.run(self.data_iter.string_handle())


class NoisyOODData:
    def __init__(self, PAD_ID, files, noisy_rate, char_list, char_probs, seed=530,
                 shuffle_buffer=-1, num_parallel_calls=32, repeat=False, prefetch=200, convert_fun=None):
        """
        Generate OOD data by adding noise to the data contained in files (a list of tfreocrd file).
        must provide file names in when creating obj. char_list and file names are written to the graph.
        This way is more flexible, allow to use numpy and python utility
        the SEED value is only useful to the shuffling opt. seed can not control noisy opt. because of
        the parallel computation (if num_paraller_calls=1, then the seed can control the noisy opt.)

        :param PAD_ID: Pad id
        :param files: a list of tfrecord file
        :param noisy_rate: rate of noise to add
        :param char_list: candidate char list for the noise
        :param char_probs: the probability for each char to appear in the noise
        :param shuffle_buffer: -1 means no shuffling, any positive value indicates the buffer for persudo shuffle.
        :param num_parallel_calls: number of parallel call for each operation
        :param repeat: True, repeat unlimited time. any positive int value n means repeat n times. False = 1
        :param prefetch: How many example to prefetch
        :param convert_fun: processing function for each example before padding and batching
        """
        padded_shape = {"utter": [None], "intent": [None], "len": [None], "id": [None]}
        padding_value = {"utter": tf.constant(PAD_ID, tf.int32), "intent": tf.constant(-1, tf.int32),
                         "len": tf.constant(-1, tf.int32), "id": tf.constant(-1, tf.int32)}
        self.noisy_rate = noisy_rate
        self.PAD_ID = PAD_ID
        self.char_list = char_list
        self.char_probs = char_probs
        self.files = files
        self.seed = seed
        self.rand_state = np.random.RandomState(seed=seed + 10)
        self.bs = tf.placeholder(tf.int64, shape=[])

        self.data_set = tf.data.TFRecordDataset(self.files)
        self.data_set = self.data_set.map(_parse_tfrecord, num_parallel_calls=num_parallel_calls)
        if convert_fun:
            self.data_set = self.data_set.map(convert_fun, num_parallel_calls=num_parallel_calls)

        if shuffle_buffer > 0:
            self.data_set = self.data_set.shuffle(buffer_size=shuffle_buffer, seed=self.seed)

        self.data_set = self.data_set.padded_batch(self.bs, padded_shapes=padded_shape, padding_values=padding_value)
        self.data_set = self.data_set.map(_suqeeze, num_parallel_calls=num_parallel_calls)

        if repeat is True:
            self.data_set = self.data_set.repeat()
        elif type(repeat) is int:
            self.data_set = self.data_set.repeat(repeat)

        self.data_set = self.data_set.map(
            lambda example: tf.py_func(self._noisy_batch, [example['utter'], example['len'], example['id'], example['intent'],
                                                           self.noisy_rate, self.char_list, self.char_probs],
                                       [tf.int32, tf.int32, tf.int32, tf.int32], stateful=False),
            num_parallel_calls=num_parallel_calls)

        self.data_set = self.data_set.map(lambda utter, length, id, intent: {"utter": utter, 'len': length, 'id': id, 'intent': intent},
                                          num_parallel_calls=num_parallel_calls)
        if prefetch > 0:
            self.data_set = self.data_set.prefetch(buffer_size=prefetch)

        self.data_iter = self.data_set.make_initializable_iterator()
        self.is_init = False

    def init(self, sess, batch_size):
        """
        initialize the dataset, only batch size is needed
        :param sess: tf session obj.
        :param batch_size: batch size for each input
        :return: None
        """
        self.is_init = True
        sess.run(self.data_iter.initializer, feed_dict={self.bs: batch_size})

    def get_handler(self, sess):
        """
        return data handler, which can be feed into some graph
        must call init before calling this function
        :param sess: tf session
        :return: a string handle
        """
        assert self.init
        return sess.run(self.data_iter.string_handle())

    def _noisy_batch(self, utter, len, id, intent, noisy_rate, char_list, char_probs):
        """
        add noise to batch
        """
        bs = utter.shape[0]
        max_len = utter.shape[1]
        mask1 = np.tile(np.arange(max_len), [bs, 1])
        mask2 = np.tile(len.reshape([-1, 1]) - 1, [1, max_len])
        mask = mask1 < mask2
        mask = np.logical_and(mask, self.rand_state.random_sample(utter.shape) < noisy_rate)
        mask = np.where(mask)
        noisy = self.rand_state.choice(char_list, size=mask[0].shape, replace=True, p=char_probs)
        utter[mask] = noisy
        return utter, len, id, intent


class RandOODData:
    def __init__(self, PAD_ID, EOS_ID, max_len, char_list, char_probs, seed=530, num_parallel_calls=32,
                 prefetch=200, convert_fun=None):
        """
        the SEED value is only useful when num_paraller_calls=1 because
        the parallel computation will ruin the function of seed
        :param PAD_ID:
        :param EOS_ID:
        :param max_len:
        :param char_list:
        :param char_probs:
        :param seed:
        :param num_parallel_calls:
        :param prefetch:
        :param convert_fun:
        """
        self.PAD_ID = PAD_ID
        self.EOS_ID = EOS_ID
        self.max_len = max_len
        self.char_list = char_list
        self.char_probs = char_probs
        self.seed = seed=530
        self.rand_state = np.random.RandomState(self.seed)
        self.bs = tf.placeholder(tf.int64, shape=[])
        not_used = tf.constant([1], dtype=tf.int32)
        self.data_set = tf.data.Dataset.from_tensors(not_used)

        self.data_set = self.data_set.repeat()
        self.data_set = self.data_set.map(
            lambda length: tf.py_func(self._gen_rand, [self.EOS_ID, self.PAD_ID, self.bs, self.max_len,
                                                       self.char_list, self.char_probs],
                                      [tf.int32, tf.int32, tf.int32, tf.int32], stateful=False),
            num_parallel_calls=num_parallel_calls)

        self.data_set = self.data_set.map(lambda utter, length, id, intent: {"utter": utter, 'len': length, 'intent': intent, 'id': id},
                                          num_parallel_calls=num_parallel_calls)
        if convert_fun:
            self.data_set = self.data_set.map(convert_fun, num_parallel_calls=num_parallel_calls)

        if prefetch > 0:
            self.data_set = self.data_set.prefetch(buffer_size=prefetch)

        self.data_iter = self.data_set.make_initializable_iterator()
        self.is_init = False

    def init(self, sess, batch_size):
        self.is_init = True
        sess.run(self.data_iter.initializer, feed_dict={self.bs: batch_size})

    def get_handler(self, sess):
        assert self.init
        return sess.run(self.data_iter.string_handle())

    def _gen_rand(self, EOS_ID, PAD_ID, bs, max_len, char_list, char_probs):
        length = self.rand_state.randint(1, max_len, [bs], dtype=np.int32)
        max_len = np.max(length)
        utter = np.ones([bs, max_len], dtype=np.int32) * PAD_ID
        mask1 = np.tile(np.arange(max_len), [bs, 1])
        mask2 = np.tile(length.reshape([-1, 1]) - 1, [1, max_len])
        mask = mask1 < mask2
        mask = np.where(mask)
        utter[mask] = self.rand_state.choice(char_list, size=mask[0].shape, replace=True, p=char_probs)
        utter[np.arange(bs), length - 1] = EOS_ID
        return utter, length, np.ones([bs], dtype=np.int32), np.zeros_like(length) - 1

