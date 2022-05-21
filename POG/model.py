import tensorflow as tf
import os


class CGAN(object):
    def __init__(self, config=None, go_id=None, eos_id=None, embed=None):
        self.config = config
        self.go_id = go_id
        self.eos_id = eos_id
        # initialize the training process
        with tf.name_scope('inputs'):
            self.data_handler = tf.placeholder(tf.string, shape=[], name='dataset_handler')
            iterator = tf.data.Iterator.from_string_handle(
                self.data_handler, {"utter": tf.int32, "id": tf.int32, "len": tf.int32, "intent": tf.int32},
                output_shapes={"utter": [None, None], "id": [None], "len": [None], "intent": [None]})
            data = iterator.get_next(name='data_input')
            self.intents = data['intent']
            self.utter = data['utter']
            self.utter_len = data['len']
            self.utter_ids = data['id']

        with tf.name_scope('common_variables'):
            self.ae_lr = tf.Variable(float(self.config['ae_lr']), trainable=False, dtype=tf.float32, name='ae_lr')
            self.g_lr = tf.Variable(float(self.config['g_lr']), trainable=False, dtype=tf.float32, name='g_lr')
            self.d_lr = tf.Variable(float(self.config['d_lr']), trainable=False, dtype=tf.float32, name='d_lr')
            self.cls_lr = tf.Variable(float(self.config['cls_lr']), trainable=False, dtype=tf.float32, name='cls_lr')

            self.ae_step = tf.Variable(0, trainable=False, name='ae_step')
            self.g_step = tf.Variable(0, trainable=False, name='g_step')
            self.d_step = tf.Variable(0, trainable=False, name='d_step')
            self.cls_step = tf.Variable(0, trainable=False, name='cls_step')
            self.keep_rate = tf.placeholder(tf.float32, shape=[], name='keep_rate')
            self.sample_t = tf.placeholder_with_default(1.0, shape=[], name='sample_t')
            # self.is_train = tf.placeholder(tf.bool, shape=[], name='train_flag')
            self.uniform_entroy = tf.log(tf.constant(self.config['intent_num'], dtype=tf.float32))
            self.max_utter_len = tf.reduce_max(self.utter_len)
            self.KL_weight = tf.minimum(1.0, tf.to_float(self.ae_step) / config['full_kl_step'])
            self.sample_rate = tf.minimum(1.0, tf.to_float(self.ae_step) / config['full_sample_step'])
            self.temp4cls = config['min_temp'] + (1 - tf.minimum(1.0, tf.to_float(
                self.ae_step) / config['full_temp_step'])) * (config['max_temp'] - config['min_temp'])

        with tf.name_scope("embedding"):
            # build the embedding table and embedding input
            if embed is None:
                # initialize the embedding randomly
                self.embed = tf.get_variable('embed', [self.config['word_vocab_size'],
                                                       self.config['word_embed_size']], tf.float32,
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))
            else:
                # initialize the embedding by pre-trained word vectors
                self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
            # self.intent_embed = tf.get_variable(
            #     'intent_embed', shape=[config['intent_num'], config['intent_embed_size']], dtype=tf.float32)

            self.utter_resp = tf.nn.embedding_lookup(self.embed, self.utter)   # [bs, utter_len, embed_size]
            self.utter_resp = tf.nn.dropout(self.utter_resp, keep_prob=self.keep_rate, name='dropped_utter_resp')
            self.batch_size = tf.shape(self.utter_resp)[0]

            self.dec_input = tf.split(self.utter, [self.max_utter_len - 1, config['max_utter_len'] - self.max_utter_len + 1], 1)[0]   # without <EOS>
            self.dec_target = tf.split(self.utter, [1, self.max_utter_len - 1, config['max_utter_len'] - self.max_utter_len], 1)[1]   # without <GO>
            self.dec_input_resp = tf.nn.embedding_lookup(self.embed, self.dec_input)
            # self.intents_resp = tf.nn.embedding_lookup(self.intent_embed, self.intents)

        with tf.variable_scope("ea"):
            with tf.variable_scope("encoder"):
                self.encoder = self._get_rnn_cell(self.config['rnn_size'], self.config['keep_rate'],
                                                  self.config['num_enc_layers'])
                _, self.enc_h = tf.nn.dynamic_rnn(cell=self.encoder, inputs=self.utter_resp,
                                                  sequence_length=self.utter_len - 1, dtype=tf.float32)

                self.enc_h = [tf.concat((self.enc_h[i].c, self.enc_h[i].h), 1) for i in range(config['num_enc_layers'])]
                # self.enc_h.append(self.intents_resp)
                self.enc_h = tf.concat(self.enc_h, 1)   # [bs, h_size]
                self.Enc_out_layer = tf.layers.Dense(self.config['latent_size'], name='enc2latent', activation=None)
                self.enc_h = self.Enc_out_layer(self.enc_h)
                self.enc_param = tf.trainable_variables()
                curr_param = tf.trainable_variables()

            with tf.variable_scope("decoder"):
                self.decoder = self._get_rnn_cell(self.config['rnn_size'], self.config['keep_rate'],
                                                  self.config['num_dec_layers'])
                self.output_layer = tf.layers.Dense(config['word_vocab_size'], use_bias=True, name='dec_output_layer',
                                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.dec_init_layer = tf.layers.Dense(config['rnn_size'] * 2, use_bias=True, activation=None,
                                                      name='dec_state_init')
                # add noise to the decoding process to smooth the latent space
                self.noisy_enc_h = self.enc_h + config['decode_noise'] * tf.random_normal(tf.shape(self.enc_h))
                self.dec_init = self._get_dec_state(self.noisy_enc_h, self.dec_init_layer, config['num_dec_layers'])
                # train_helper = tf.contrib.seq2seq.TrainingHelper(self.dec_input_resp, self.utter_len - 1)
                train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    self.dec_input_resp, self.utter_len - 1, self.embed, self.sample_rate)
                train_decoder = tf.contrib.seq2seq.BasicDecoder(
                    self.decoder, train_helper, self.dec_init, self.output_layer)
                train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(train_decoder, impute_finished=True)
                self.decoder_logits = train_outputs.rnn_output
                self.dec_param = [self.embed]
                for i in tf.trainable_variables():
                    if i not in curr_param:
                        self.dec_param.append(i)
                curr_param = tf.trainable_variables()

        with tf.variable_scope("G"):
            self.G_fn = []
            for i, size in enumerate(list(map(int, self.config['g_size'].split(',')))):
                self.G_fn.append(tf.layers.Dense(size, name='G_l{}_{}'.format(i, size)))
                self.G_fn.append(tf.nn.leaky_relu)
                # self.G_fn.append(('G_layer_norm' + str(i), tf.contrib.layers.layer_norm))
                # self.G_fn.append(tf.layers.BatchNormalization(name='G_bn{}'.format(i)))

            self.G_fn.append(tf.layers.Dense(self.config['latent_size'], activation=None, name='G_out'))

            self.noise = tf.random_normal(
                shape=[self.batch_size, self.config['latent_size']], dtype=tf.float32, name='G_rand')

            # self.intent_sample = tf.random_uniform(
            # shape=[self.batch_size], minval=1, maxval=config['intent_num'] - 1, dtype=tf.int32)
            # self.intent_sample_resp = tf.nn.embedding_lookup(self.intent_embed, self.intent_sample)
            # self.sample_generator_out = self._apply_fn_seq(
            # self.G_fn, tf.concat((self.noise, self.intent_sample_resp), axis=1)) # [bs, latent_size]
            # self.feed_generator_out = self._apply_fn_seq(self.G_fn, tf.concat((self.noise, self.intents_resp),axis=1))
            self.generator_out = self._apply_fn_seq(self.G_fn, self.noise)

            self.fake_dec_init = self._get_dec_state(self.generator_out, self.dec_init_layer, config['num_dec_layers'])
            # train_helper = tf.contrib.seq2seq.TrainingHelper(self.dec_input_resp, self.utter_len - 1)
            g_train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                self.dec_input_resp, self.utter_len - 1, self.embed, 1.0)
            g_train_decoder = tf.contrib.seq2seq.BasicDecoder(
                self.decoder, g_train_helper, self.fake_dec_init, self.output_layer)
            self.g_train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(g_train_decoder, impute_finished=True)
            self.g_decode_logits = self.g_train_outputs.rnn_output   # [bs, vocab_size]

            # self.G_param = [self.intent_embed]
            self.G_param = []
            for i in tf.trainable_variables():
                if i not in curr_param:
                    self.G_param.append(i)
            curr_param = tf.trainable_variables()

        with tf.variable_scope('D'):
            self.D_fn = []
            for i, size in enumerate(map(int, self.config['d_size'].split(','))):
                self.D_fn.append(tf.layers.Dense(size, name='D_l{}_{}'.format(i, size)))
                self.D_fn.append(tf.nn.leaky_relu)
                # no BN for the first layer
                # if i != 0:
                #     self.D_fn.append(('D_layer_norm' + str(i), tf.contrib.layers.layer_norm))
                #     self.D_fn.append(tf.layers.BatchNormalization(name='D_bn{}'.format(i)))

            self.D_fn.append(tf.layers.Dense(1, activation=None, name='D_out'))

            # self.real_D_logits = self._apply_fn_seq(
            # self.D_fn, tf.concat((self.enc_h, self.intents_resp), axis=1))    # [bs, 1]
            # self.sample_fake_D_logits = self._apply_fn_seq(
            # self.D_fn, tf.concat((self.sample_generator_out, self.intent_sample_resp), axis=1))
            # self.feed_fake_D_logits = self._apply_fn_seq(
            # self.D_fn, tf.concat((self.feed_generator_out, self.intents_resp), axis=1))

            self.real_D_logits = self._apply_fn_seq(self.D_fn, self.enc_h)    # [bs, 1]
            self.fake_D_logits = self._apply_fn_seq(self.D_fn, self.generator_out)

            # self.fake_D_logits = self._apply_fn_seq(self.D_fn, self.generator_out)    # [bs, 1]
            self.D_param = []
            for i in tf.trainable_variables():
                if i not in curr_param:
                    self.D_param.append(i)
            curr_param = tf.trainable_variables()

        with tf.variable_scope("cls"):
            kernel_size = list(map(int, config['kernel_sizes'].split(',')))
            cnn_features = list(map(int, config['cnn_features'].split(',')))
            cls_mlp_size = list(map(int, config['cls_mlp_size'].split(',')))
            self.decoder_softmax = tf.nn.softmax(self.decoder_logits / self.temp4cls)   # [bs, len, vocab_size]
            self.real_cls_input = tf.tensordot(self.decoder_softmax, self.embed, axes=[[2], [0]])    # [bs, len, embed_size]
            self.real_cls_logits = self._cnn(
                self.real_cls_input, self.keep_rate, kernel_size, cnn_features, cls_mlp_size, config['intent_num'])

        with tf.variable_scope("cls", reuse=True):
            self.g_softmax = tf.nn.softmax(self.g_decode_logits / self.temp4cls)
            self.fake_cls_input = tf.tensordot(self.g_softmax, self.embed, axes=[[2], [0]])
            self.fake_cls_logits = self._cnn(
                self.fake_cls_input, self.keep_rate, kernel_size, cnn_features, cls_mlp_size, config['intent_num'])

            self.cls_param = []
            for i in tf.trainable_variables():
                if i not in curr_param:
                    self.cls_param.append(i)

        with tf.variable_scope("cls", reuse=True):
            self.cls_infer_resp = tf.nn.embedding_lookup(self.embed, self.dec_target)    # [bs, len, embed_size]
            self.cls_infer_resp = self.cls_infer_resp * (tf.reshape(tf.sequence_mask(
                self.utter_len - 1, self.max_utter_len - 1, dtype=tf.float32), [-1, self.max_utter_len - 1, 1]))

            self.cls_infer_logits = self._cnn(
                self.cls_infer_resp, self.keep_rate, kernel_size, cnn_features, cls_mlp_size, config['intent_num'])
            self.cls_infer_pred = tf.argmax(self.cls_infer_logits, 1, output_type=tf.int32)
            self.cls_infer_acc = tf.reduce_mean(tf.cast(tf.equal(self.cls_infer_pred, self.intents), tf.float32))

        with tf.name_scope("losses"):
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.decoder_logits,
                                                                      labels=self.dec_target)
            decoder_mask = tf.sequence_mask(self.utter_len - 1, self.max_utter_len - 1, dtype=tf.float32)

            self.reconstruct_loss = tf.reduce_sum(crossent * decoder_mask) / tf.cast(self.batch_size, tf.float32)
            if config['fix_ae']:
                self.reconstruct_loss = (1 - self.KL_weight) * self.reconstruct_loss
            self.ppl_loss = tf.reduce_sum(crossent * decoder_mask) / tf.reduce_sum(decoder_mask)

            self.real_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.real_cls_logits, labels=self.intents))
            self.real_cls_loss = self.KL_weight * self.real_cls_loss    # annealing

            self.fake_cls_kl_loss = - tf.reduce_mean(self.fake_cls_logits, axis=1) +\
                                    tf.reduce_logsumexp(self.fake_cls_logits, axis=1)
            self.fake_cls_kl_loss = tf.reduce_mean(
                self.fake_cls_kl_loss - tf.ones_like(self.fake_cls_kl_loss) * self.uniform_entroy)
            self.fake_cls_kl_loss = self.KL_weight * self.fake_cls_kl_loss    # annealing

            self.real_D_loss = tf.reduce_mean(self.real_D_logits)
            self.real_D_loss = self.KL_weight * self.real_D_loss   # annealing
            self.fake_D_loss = tf.reduce_mean(self.fake_D_logits)
            self.fake_D_loss = self.KL_weight * self.fake_D_loss   # annealing

            real_intent_pred = tf.argmax(self.real_cls_logits, 1, output_type=tf.int32)
            fake_intent_pred = tf.argmax(self.fake_cls_logits, 1, output_type=tf.int32)
            self.real_cls_acc = tf.reduce_mean(tf.cast(tf.equal(real_intent_pred, self.intents), tf.float32))
            self.fake_cls_acc = tf.reduce_mean(tf.cast(tf.equal(fake_intent_pred, self.intents), tf.float32))
            self.real_D_acc = tf.reduce_mean(
                tf.cast(tf.less(self.real_D_logits, tf.zeros_like(self.real_D_logits)), tf.float32))
            self.fake_D_acc = tf.reduce_mean(
                tf.cast(tf.greater(self.fake_D_logits, tf.zeros_like(self.fake_D_logits)), tf.float32))

            with tf.name_scope('gradient_penalty'), tf.variable_scope('D'):
                # steal from ARAE code
                alpha = tf.random_uniform([self.batch_size, 1], minval=0, maxval=1)
                interpolates = alpha * self.enc_h + (1 - alpha) * self.generator_out
                d_logits_gp = self._apply_fn_seq(self.D_fn, interpolates)
                gp_gradients = tf.gradients(d_logits_gp, interpolates)[0]
                ddx = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis=1))
                self.gradient_penalty = tf.reduce_mean(tf.square(ddx - 1.0) * config['gp_lambda'])
                # self.gradient_penalty = tf.reduce_mean(
                #     (tf.norm(gp_gradients, ord=2, axis=1) - 1) ** 2) * config['gp_lambda']
                self.gradient_penalty = self.KL_weight * self.gradient_penalty   # annealing

            self.ae_scalar_summ_op = tf.summary.merge([
                tf.summary.scalar("loss/ae_ppl_loss", tf.exp(self.ppl_loss), family='ae'),
                tf.summary.scalar("loss/kl_weight", self.KL_weight, family='ae'),
                tf.summary.scalar("loss/temp_cls", self.temp4cls, family='ae'),
                tf.summary.scalar("loss/sample_ratio", self.sample_rate, family='ae')
            ])
            # self.g_scalar_summ_op = tf.summary.merge([
            #     tf.summary.scalar("g_out_var", self.gen_out_variance, family='g'),
            #     tf.summary.scalar("g_out_noise_var", self.noise_variance, family='g')
            # ])
            self.d_scalar_summ_op = tf.summary.merge([
                tf.summary.scalar("loss/real_D_loss", self.real_D_loss, family='d'),
                tf.summary.scalar("loss/fake_D_loss", self.fake_D_loss, family='d'),
                tf.summary.scalar("acc/real_D_acc", self.real_D_acc, family='d'),
                tf.summary.scalar("acc/fake_D_acc", self.fake_D_acc, family='d'),
                tf.summary.scalar("loss/gradient_penalty", self.gradient_penalty, family='d')
            ])
            self.cls_scalar_summ_op = tf.summary.merge([
                tf.summary.scalar("loss/real_cls_loss", self.real_cls_loss, family='cls'),
                tf.summary.scalar("loss/fake_cls_kl_loss", self.fake_cls_kl_loss, family='cls'),
                tf.summary.scalar("acc/real_cls_acc", self.real_cls_acc, family='cls'),
                tf.summary.scalar("acc/fake_cls_acc", self.fake_cls_acc, family='cls')
            ])

        with tf.name_scope('optimizer'):
            # AE optimizer
            # tf.train.GradientDescentOptimizer(self.ae_lr)
            self.ae_opt = tf.train.AdamOptimizer(self.ae_lr)
            self.g_opt = tf.train.AdamOptimizer(self.g_lr)
            self.d_opt = tf.train.AdamOptimizer(self.d_lr)
            self.cls_opt = tf.train.AdamOptimizer(self.cls_lr)

            # train AE
            ae_param = list(set(self.enc_param + self.dec_param))
            ae_gradients = tf.gradients(self.reconstruct_loss, ae_param)
            ae_gradients, _ = tf.clip_by_global_norm(ae_gradients, 1.0)
            self.ae_grad_summ_op = tf.summary.merge([tf.summary.histogram(
                "%s-grad" % g[1].name, g[0], family='ae_recons') for g in zip(ae_gradients, ae_param)])
            self.train_ae = self.ae_opt.apply_gradients(
                zip(ae_gradients, ae_param), global_step=self.ae_step, name='ae_train_op')

            # train D
            d_grads = self.d_opt.compute_gradients(self.real_D_loss - self.fake_D_loss + self.gradient_penalty, var_list=self.D_param)
            self.train_d = self.d_opt.apply_gradients(d_grads, global_step=self.d_step, name='d_train_op')
            self.d_grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0], family='d') for g in d_grads])
            # self.train_d = self.d_opt.minimize(
            #     self.real_D_loss - self.fake_D_loss + self.gradient_penalty, var_list=self.D_param,
            #     global_step=self.d_step, name='d_train_op')

            # train G to fool D (enforce G to produce real latent)
            g_grads = self.g_opt.compute_gradients(self.fake_D_loss, var_list=self.G_param)
            self.train_g = self.g_opt.apply_gradients(g_grads, global_step=self.g_step, name='g_train_op')
            self.g_grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0], family='g') for g in g_grads])
            # self.train_g = self.g_opt.minimize(
            #     self.fake_D_loss, var_list=self.G_param, global_step=self.g_step, name='g_train_op')

            # params for enc (enforce enc to produce fake latent)
            d_into_ae_grads = tf.gradients(-self.real_D_loss, self.enc_param)
            d_into_ae_grads, _ = tf.clip_by_global_norm(d_into_ae_grads, 1.0)
            self.d_into_ae_grad_summ_op = tf.summary.merge([tf.summary.histogram(
                "%s-grad" % g[1].name, g[0], family='d_into_ae') for g in zip(d_into_ae_grads, self.enc_param)])
            self.train_d_into_ae = self.ae_opt.apply_gradients(
                zip(d_into_ae_grads, self.enc_param), name='d_to_ae_train_op')

            # train cls
            # cls_enc_param = list(set(self.enc_param + self.cls_param))
            # cls_enc_grads = tf.gradients(self.real_cls_loss, cls_enc_param)
            # cls_enc_grads, _ = tf.clip_by_global_norm(cls_enc_grads, 1.0)
            # self.train_cls = self.cls_opt.apply_gradients(
            #     zip(cls_enc_grads, cls_enc_param), global_step=self.cls_step, name='cls_train_op')
            cls_grads = self.cls_opt.compute_gradients(self.real_cls_loss, var_list=self.cls_param)
            self.train_cls = self.cls_opt.apply_gradients(cls_grads, global_step=self.cls_step, name='cls_train_op')
            self.cls_grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0], family='cls') for g in cls_grads])
            # self.train_cls = self.cls_opt.minimize(
            #     self.real_cls_loss, var_list=self.cls_param, global_step=self.cls_step, name='cls_train_op')

            # train G to produce un-labeled latent
            regul_g_grads = self.g_opt.compute_gradients(self.fake_cls_kl_loss, var_list=self.G_param)
            self.train_regularized_g = self.g_opt.apply_gradients(regul_g_grads, global_step=self.g_step, name='regularized_g_train_op')
            self.regul_g_grad_summ_op = tf.summary.merge([tf.summary.histogram("%s-grad" % g[1].name, g[0], family='regu_g') for g in regul_g_grads])
            self.train_regularized_g = self.g_opt.minimize(
                self.fake_cls_kl_loss, var_list=self.G_param, global_step=self.g_step, name='regularized_g_train_op')

        with tf.name_scope("infer"):
            with tf.name_scope("ae_noise_greedy"):
                infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.embed, tf.fill([self.batch_size], self.go_id), self.eos_id)
                infer_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder, infer_helper, self.dec_init, self.output_layer)
                infer_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    infer_decoder, impute_finished=True, maximum_iterations=config['max_decode_len'])
                self.infer_noise_greedy_utter = infer_output.sample_id   # [bs, max_len]

                clear_dec_init = self._get_dec_state(self.enc_h, self.dec_init_layer, config['num_dec_layers'])
                clear_infer_decoder = tf.contrib.seq2seq.BasicDecoder(self.decoder, infer_helper, clear_dec_init, self.output_layer)
                clear_infer_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    clear_infer_decoder, impute_finished=True, maximum_iterations=config['max_decode_len'])
                self.infer_clear_greedy_utter = clear_infer_output.sample_id   # [bs, max_len]

            with tf.name_scope("ae_beam"):
                tiled_infer_init = tf.contrib.seq2seq.tile_batch(self.dec_init, multiplier=config['beam_width'])
                beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=self.decoder, embedding=self.embed, initial_state=tiled_infer_init, output_layer=self.output_layer,
                    start_tokens=tf.fill([self.batch_size], self.go_id), end_token=self.eos_id,
                    beam_width=config['beam_width'])
                beam_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    beam_decoder, impute_finished=False, maximum_iterations=config['max_decode_len'])
                self.beam_utter = beam_output.predicted_ids    # [bs, max_len, beam_width]

            with tf.name_scope("greedy_latent"):
                dec_init_from_g = self._get_dec_state(self.generator_out, self.dec_init_layer, config['num_dec_layers'])
                latent_decoder = tf.contrib.seq2seq.BasicDecoder(
                    self.decoder, infer_helper, dec_init_from_g, self.output_layer)
                latent_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    latent_decoder, impute_finished=True, maximum_iterations=config['max_decode_len'])
                self.latent_greedy_utter = latent_output.sample_id

            with tf.name_scope("sample_latent"):
                latent_sample_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                    self.embed, tf.fill([self.batch_size], self.go_id), self.eos_id, softmax_temperature=self.sample_t)
                latent_sample_decoder = tf.contrib.seq2seq.BasicDecoder(
                    self.decoder, latent_sample_helper, dec_init_from_g, self.output_layer)
                latent_sample_out, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=latent_sample_decoder, maximum_iterations=config['max_decode_len'])
                self.latent_sample_utter = latent_sample_out.sample_id

            with tf.name_scope("greedy_noise"):
                noise = tf.random_normal(shape=[self.batch_size, self.config['latent_size']], dtype=tf.float32)
                dec_init_from_noise = self._get_dec_state(noise, self.dec_init_layer, config['num_dec_layers'])
                noise_decoder = tf.contrib.seq2seq.BasicDecoder(
                    self.decoder, infer_helper, dec_init_from_noise, self.output_layer)
                noise_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    noise_decoder, impute_finished=True, maximum_iterations=config['max_decode_len'])
                self.noise_greedy_utter = noise_output.sample_id

        with tf.name_scope("saver"):
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, max_to_keep=50,
                                        pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
            self.best_saver = tf.train.Saver(max_to_keep=5, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
            # self.summary_op = tf.summary.merge_all()

        print('---------trainable_variables-----------')
        for var in tf.trainable_variables():
            in_info = 'in '
            if var in self.enc_param:
                in_info += 'enc '
            if var in self.dec_param:
                in_info += 'dec '
            if var in self.G_param:
                in_info += 'G '
            if var in self.D_param:
                in_info += 'D '
            if var in self.cls_param:
                in_info += 'cls '
            print(var, in_info)

    @staticmethod
    def _cnn(input_resp, keep_ratio, k_size, feature_sizes, mlp_size, intent_num):
        assert len(k_size) == len(feature_sizes)
        features = []
        count = 0
        for ks, num_features in zip(k_size, feature_sizes):
            conv = tf.layers.conv1d(input_resp, num_features, ks, name='conv_{}_ks_{}'.format(count, ks), padding='same')
            pooling = tf.reduce_max(conv, axis=1, name='max_pooling_{}_ks_{}'.format(count, ks))
            features.append(pooling)
            count += 1
        features = tf.concat(features, axis=1, name='cnn_feature')
        for i in mlp_size:
            features = tf.layers.dense(features, i, activation=tf.nn.tanh)
            features = tf.nn.dropout(features, keep_prob=keep_ratio)
        logits = tf.layers.dense(features, intent_num)
        return logits

    @staticmethod
    def _get_rnn_cell(size, keep_rate, layer_num):
        return tf.contrib.rnn.MultiRNNCell(
            [tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(size), variational_recurrent=True,
                dtype=tf.float32, output_keep_prob=keep_rate, state_keep_prob=keep_rate)) for _ in range(layer_num)])

    @staticmethod
    def _apply_fn_seq(fn_seq, inputs):
        # tf.get_variable_scope().reuse_variables()
        for fn in fn_seq:
            if type(fn) is tuple:
                inputs = fn[1](inputs, reuse=tf.AUTO_REUSE, scope=fn[0])
            else:
                inputs = fn(inputs)
        return inputs

    @staticmethod
    def _get_dec_state(state, dec_init_fn, layers):
        state = dec_init_fn(state)
        return tuple([tf.nn.rnn_cell.LSTMStateTuple(tf.split(state, 2, 1)[0],
                                                    tf.split(state, 2, 1)[1]) for _ in range(layers)])
