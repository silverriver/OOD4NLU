import tensorflow as tf
import utils


class ICM:
    def __init__(self, config, embed=None):
        self.config = config
        self.seed = self.config['seed']
        tf.set_random_seed(self.seed)
        self.lr = tf.Variable(float(self.config['learning_rate']), trainable=False,
                              dtype=tf.float32, name='learning_rate')
        self.lr_decay_op = self.lr.assign(self.lr * self.config['lr_decay_factor'])
        self.global_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='global_step')
        self.keep_rate = tf.placeholder(tf.float32, shape=None, name='keep_rate')
        self.uniform_entroy = tf.log(tf.constant(self.config['intent_num'], dtype=tf.float32))

        self.ind_data_handler = tf.placeholder(tf.string, shape=[], name='ind_dataset_handler')
        self.ood_data_handler = tf.placeholder(tf.string, shape=[], name='ood_dataset_handler')
        self.inputs_type = {"utter": tf.int32, "len": tf.int32, "intent": tf.int32, "id": tf.int32}
        self.inputs_shape = {"utter": [None, None], "len": [None], "intent": [None], "id": [None]}
        self.ind_iter = tf.data.Iterator.from_string_handle(self.ind_data_handler, self.inputs_type, self.inputs_shape)
        self.ood_iter = tf.data.Iterator.from_string_handle(self.ood_data_handler, self.inputs_type, self.inputs_shape)

        # with tf.device('/cpu:0'):
        with tf.name_scope('Input'):
            self.ind_data = self.ind_iter.get_next(name='ind_data')
            self.ood_data = self.ood_iter.get_next(name='ood_data')
            self.ind_utt = self.ind_data['utter']  # [bs, max_l]
            self.ood_utt = self.ood_data['utter']  # [bs, max_l]
            self.intent = self.ind_data['intent']  # [bs]
            self.ind_ids = self.ind_data['id']         # [bs]
            self.ood_ids = self.ood_data['id']         # [bs]
            self.ind_bs = tf.shape(self.ind_utt)[0]
            self.ood_bs = tf.shape(self.ood_utt)[0]

            self.char_embed = self._char_embed_layer(embed)   # [char_vocab, char_embed]
            # self.state_embed = self._state_embed_layer()

            # [bs, max_l, char_embed]
            self.input_utt = tf.nn.embedding_lookup(self.char_embed, self.ind_utt, name='char_lookup_ind')
            self.input_utt = tf.nn.dropout(self.input_utt, self.keep_rate, name='char_dropout_ind')

            # [bs, max_l, char_embed]
            self.input_utt_ood = tf.nn.embedding_lookup(self.char_embed, self.ood_utt, name='char_lookup_ood')
            self.input_utt_ood = tf.nn.dropout(self.input_utt_ood, self.keep_rate, name='char_dropout_ood')

        with tf.name_scope('CNN'):
            # [bs, hs]
            with tf.variable_scope('CNN'):
                self.cnn_features_ind = self._cnn(self.input_utt)
            with tf.variable_scope('CNN', reuse=True):
                self.cnn_features_ood = self._cnn(self.input_utt_ood)

        with tf.name_scope('Hidden'):
            # [bs, hs]
            with tf.variable_scope('Hidden'):
                self.hidden_features_ind = self._dropout_hidden(self.cnn_features_ind, self.config['hidden_sizes'],
                                                                keep_rate=self.keep_rate)
            with tf.variable_scope('Hidden', reuse=True):
                self.hidden_features_ood = self._dropout_hidden(self.cnn_features_ood, self.config['hidden_sizes'],
                                                                keep_rate=self.keep_rate)

        with tf.name_scope('Output'):
            with tf.variable_scope('logits'):
                self.intent_logits_ind = tf.layers.dense(self.hidden_features_ind, self.config['intent_num'],
                                                         name='fc_intent_logit')
            with tf.variable_scope('logits', reuse=True):
                # [bs, intent_num]
                self.intent_logits_ood = tf.layers.dense(self.hidden_features_ood, self.config['intent_num'],
                                                         name='fc_intent_logit')
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.intent_logits_ind,
                                                                                labels=self.intent,
                                                                                name='ce_layer')
            self.intent_loss = tf.reduce_mean(self.cross_entropy, name='intent_loss')
            self.intent_softmax = tf.nn.softmax(self.intent_logits_ind, name='intent_softmax')   # [bs, intent_num]
            self.intent_predict = tf.argmax(self.intent_softmax, 1, name='predict_intent', output_type=tf.int32)  # [bs]

            correct_predict = tf.equal(self.intent_predict, self.intent)
            self.acc = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name='acc')

            with tf.name_scope('KL'):
                # [bs]
                self.kl_term = - tf.reduce_mean(self.intent_logits_ood, axis=1) + \
                               tf.reduce_logsumexp(self.intent_logits_ood, axis=1)
                self.kl_term = self.kl_term - tf.ones_like(self.kl_term) * self.uniform_entroy
                self.kl_loss = tf.multiply(self.config['kl_loss_ratio'], tf.reduce_mean(self.kl_term), name='kl_loss')

        with tf.name_scope('Optimizer'):
            self.loss = self.intent_loss
            if self.config['use_kl_loss']:
                self.loss = self.loss + self.kl_loss

            if self.config['opt'] == 'SGD':
                self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            elif self.config['opt'] == 'Adam':
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            else:
                self.opt = None

            self.train_op = self.opt.minimize(self.loss, global_step=self.global_step)

        # print all variables
        for var in tf.trainable_variables():
            print(var)

        self.saver = tf.train.Saver(max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def train(self, sess, ind_handler, ood_handler, keep_rate):
        input_feed = {self.ind_data_handler: ind_handler, self.ood_data_handler: ood_handler, self.keep_rate: keep_rate}
        output_feed = [self.train_op, self.loss, self.acc, self.intent_loss, self.kl_loss]
        return sess.run(output_feed, feed_dict=input_feed)

    def _char_embed_layer(self, char_embed):
        with tf.variable_scope('embedding'):
            if char_embed is None:
                return tf.get_variable('char_embed_layer',
                                       [self.config['char_vocab_size'], self.config['char_embed_size']],
                                       dtype=tf.float32)
            else:
                return tf.get_variable('char_embed_layer', dtype=tf.float32, initializer=char_embed)

    def _state_embed_layer(self):
        with tf.variable_scope('embedding'):
            return tf.get_variable('state_embed_layer',
                                   [self.config['state_vocab_size'], self.config['state_vocab_size'] * 2],
                                   dtype=tf.float32)

    def _cnn(self, input_utt):
        features = []
        count = 0
        for ks, num_features in zip(self.config['kernel_sizes'], self.config['cnn_features']):
            conv = tf.layers.conv1d(input_utt, num_features, ks, name='conv_{}_ks_{}'.format(count, ks), padding='same')
            pooling = tf.reduce_max(conv, axis=1, name='max_pooling_{}_ks_{}'.format(count, ks))
            features.append(pooling)
            count += 1
        return tf.concat(features, axis=1, name='cnn_feature')

    @staticmethod
    def _dropout_hidden(input, hiddens, keep_rate):
        count = 0
        for hs in hiddens:
            input = tf.layers.dense(input, hs, activation=tf.tanh, name='fc_{}_hs_{}'.format(count, hs))
            input = tf.nn.dropout(input, keep_prob=keep_rate, name='fc_drop_{}_hs_{}'.format(count, hs))
            count += 1
        return input






