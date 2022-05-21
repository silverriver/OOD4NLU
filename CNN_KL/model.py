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
        self.ind_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='ind_step')
        self.ood_step = tf.Variable(0, trainable=False, dtype=tf.int32, name='ood_step')
        self.keep_rate = tf.placeholder(tf.float32, shape=None, name='keep_rate')
        self.uniform_entroy = tf.log(tf.constant(self.config['intent_num'], dtype=tf.float32))

        self.ind_data_handler = tf.placeholder(tf.string, shape=[], name='ind_dataset_handler')
        self.ood_data_handler = tf.placeholder(tf.string, shape=[], name='ood_dataset_handler')
        self.inputs_type = {"utter": tf.int32, "len": tf.int32, "intent": tf.int32, "id": tf.int32}
        self.inputs_shape = {"utter": [None, None], "len": [None], "intent": [None], "id": [None]}
        self.ind_data_iter = tf.data.Iterator.from_string_handle(self.ind_data_handler, self.inputs_type, self.inputs_shape)
        self.ood_data_iter = tf.data.Iterator.from_string_handle(self.ood_data_handler, self.inputs_type, self.inputs_shape)

        # with tf.device('/cpu:0'):
        with tf.name_scope('Input'):
            # read in IND data
            self.ind_data = self.ind_data_iter.get_next(name='data')
            self.ind_utt = self.ind_data['utter']  # [bs, max_l]
            self.ind_intent = self.ind_data['intent']  # [bs]
            self.ind_ids = self.ind_data['id']         # [bs]
            self.ind_bs = tf.shape(self.ind_utt)[0]

            # read in OOD data
            self.ood_data = self.ood_data_iter.get_next(name='data')
            self.ood_utt = self.ood_data['utter']  # [bs, max_l]
            self.ood_intent = self.ood_data['intent']  # [bs]
            self.ood_ids = self.ood_data['id']         # [bs]
            self.ood_bs = tf.shape(self.ood_utt)[0]

            self.char_embed = self._char_embed_layer(embed)   # [char_vocab, char_embed]

            # [bs, max_l, char_embed]
            self.ind_input_utt = tf.nn.embedding_lookup(self.char_embed, self.ind_utt, name='char_lookup_ind')
            self.ind_input_utt = tf.nn.dropout(self.ind_input_utt, self.keep_rate, name='char_dropout_ind')

            self.ood_input_utt = tf.nn.embedding_lookup(self.char_embed, self.ood_utt, name='char_lookup_ood')
            self.ood_input_utt = tf.nn.dropout(self.ood_input_utt, self.keep_rate, name='char_dropout_ood')

        with tf.name_scope('CNN'):
            # [bs, hs]
            with tf.variable_scope('CNN'):
                self.ind_cnn_features = self._cnn(self.ind_input_utt)
            with tf.variable_scope('CNN', reuse=True):
                self.ood_cnn_features = self._cnn(self.ood_input_utt)

        with tf.name_scope('Hidden'):
            # [bs, hs]
            with tf.variable_scope('Hidden'):
                self.ind_hidden_features = self._dropout_hidden(
                    self.ind_cnn_features, self.config['hidden_sizes'], keep_rate=self.keep_rate)

            with tf.variable_scope('Hidden', reuse=True):
                self.ood_hidden_features = self._dropout_hidden(
                    self.ood_cnn_features, self.config['hidden_sizes'], keep_rate=self.keep_rate)

        with tf.name_scope('Output'):
            with tf.variable_scope('logits'):
                self.ind_logits = tf.layers.dense(
                    self.ind_hidden_features, self.config['intent_num'], name='fc_intent_logit')

            with tf.variable_scope('logits', reuse=True):
                self.ood_logits = tf.layers.dense(
                    self.ood_hidden_features, self.config['intent_num'], name='fc_intent_logit')

            with tf.name_scope('IND_loss'):
                self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.ind_logits, labels=self.ind_intent, name='ce_layer')

                self.intent_loss = tf.reduce_mean(self.cross_entropy, name='intent_loss')
                self.intent_softmax = tf.nn.softmax(self.ind_logits, name='intent_softmax')   # [bs, intent_num]
                self.intent_predict = tf.argmax(self.intent_softmax, 1, name='predict_intent', output_type=tf.int32)  # [bs]

                correct_predict = tf.equal(self.intent_predict, self.ind_intent)
                self.acc = tf.reduce_mean(tf.cast(correct_predict, tf.float32), name='acc')
                tf.summary.scalar('intent_loss', self.intent_loss)
                tf.summary.scalar('acc', self.acc)

            with tf.name_scope('KL'):
                # [bs]
                self.ood_intent_softmax = tf.nn.softmax(self.ood_logits, name='ood_intent_softmax')   # [bs, intent_num]
                self.kl_term = - tf.reduce_mean(self.ood_logits, axis=1) + \
                               tf.reduce_logsumexp(self.ood_logits, axis=1)
                self.kl_term = self.kl_term - tf.ones_like(self.kl_term) * self.uniform_entroy
                self.kl_term = tf.maximum(self.config['kl_hinge'], self.kl_term)
                self.kl_loss = tf.reduce_mean(self.kl_term, name='kl_loss')
                tf.summary.scalar('kl_loss', self.kl_loss)

        with tf.name_scope('Optimizer'):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.variables = tf.trainable_variables()
            self.ind_gradient = tf.gradients(self.intent_loss, self.variables)
            self.ood_gradient = tf.gradients(tf.multiply(self.config['kl_loss_ratio'], self.kl_loss), self.variables)
            self.ind_train_op = self.opt.apply_gradients(zip(self.ind_gradient, self.variables), global_step=self.ind_step)
            self.ood_train_op = self.opt.apply_gradients(zip(self.ood_gradient, self.variables), global_step=self.ood_step)

        # print all variables
        for var in tf.trainable_variables():
            print(var)

        self.saver = tf.train.Saver(max_to_keep=3, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
        self.summ_op = tf.summary.merge_all()

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






