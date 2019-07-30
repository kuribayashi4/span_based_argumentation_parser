import chainer
import chainer.functions as chaFunc
import chainer.links as chaLink
import chainer.initializers as chaInit
import numpy as np
import h5py


class BaseArgStrParser(chainer.Chain):
    def __init__(self,
                 vocab,
                 essay_info_dict,
                 para_info_dict,
                 max_n_spans_para,
                 max_n_paras,
                 max_n_tokens,
                 settings,
                 baseline_heuristic=False,
                 use_elmo=True,
                 decoder="proposed"):

        ##########################
        # set default attributes #
        ##########################
        self.vocab = vocab
        self.essay_info_dict = essay_info_dict
        self.para_info_dict = para_info_dict
        self.encVocabSize = len(vocab)
        self.eDim = settings.eDim
        self.hDim = settings.hDim
        self.dropout = settings.dropout
        self.dropout_lstm = settings.dropout_lstm
        self.dropout_embedding = settings.dropout_embedding
        self.max_n_para = max_n_paras
        self.max_n_spans = max_n_spans_para
        self.max_n_tokens = max_n_tokens
        self.decoder = decoder

        self.args = settings

        ###############
        # Select LSTM #
        ###############
        self.lstm_ac = settings.lstm_ac
        self.lstm_shell = settings.lstm_shell
        self.lstm_ac_shell = settings.lstm_ac_shell
        self.lstm_type = settings.lstm_type

        #######################
        # position information #
        #######################
        self.position_info_size = self.max_n_spans*3
        self.relative_position_info_size = 21

        ################
        # elmo setting #
        ################
        self.use_elmo = use_elmo
        if self.use_elmo:
            self.eDim = 1024

        ##########
        # others #
        ##########
        self.baseline_heuristic = baseline_heuristic

        ##############################
        # hidden representation size #
        ##############################
        self.lstm_out = self.hDim*2

        if self.use_elmo:
            self.bow_feature_size = len(self.vocab)
        else:
            self.bow_feature_size = len(self.vocab) + 3*self.eDim

        self.bow_rep_size = self.lstm_out

        # the size of representation created with LSTM-minus
        self.span_rep_size = self.lstm_out * 2

        # output of AC layer
        if self.lstm_ac:
            self.ac_rep_size = self.lstm_out
        else:
            self.ac_rep_size = self.span_rep_size

        # output of AM layer
        if self.lstm_shell:
            self.shell_rep_size = self.lstm_out
        else:
            self.shell_rep_size = self.span_rep_size

        # the size of ADU representation
        self.ac_shell_rep_size_in = self.ac_rep_size +\
            self.shell_rep_size + self.position_info_size + self.bow_rep_size

        # output of ADU layer
        if self.lstm_ac_shell:
            self.ac_shell_rep_size_out = self.lstm_out
        else:
            self.ac_shell_rep_size_out = self.ac_shell_rep_size_in

        # output of Encoder (ADU-level)
        self.reps_for_type_classification = self.ac_shell_rep_size_out

        # the size of ADU representations for link identification
        if self.lstm_type:
            self.type_rep_size = self.lstm_out
        else:
            self.type_rep_size = self.ac_shell_rep_size_out

        # the size of ADU pair representation
        self.span_pair_size = self.type_rep_size*3 + self.relative_position_info_size

        n_ac_shell_latm_layers = 1

        super(BaseArgStrParser, self).__init__()

        with self.init_scope():
            self.Embed_x = chaLink.EmbedID(self.encVocabSize,
                                           self.eDim,
                                           ignore_label=-1)

            self.Bilstm = chaLink.NStepBiLSTM(n_layers=1,
                                              in_size=self.eDim,
                                              out_size=self.hDim,
                                              dropout=self.dropout_lstm)

            if self.lstm_ac:
                self.AcBilstm = chaLink.NStepBiLSTM(n_layers=1,
                                                    in_size=self.span_rep_size,
                                                    out_size=self.hDim,
                                                    dropout=self.dropout_lstm)
            if self.lstm_shell:
                self.ShellBilstm = chaLink.NStepBiLSTM(n_layers=1,
                                                       in_size=self.span_rep_size, 
                                                       out_size=self.hDim,
                                                       dropout=self.dropout_lstm)

            self.AcShellBilstm = chaLink.NStepBiLSTM(n_layers=n_ac_shell_latm_layers,
                                                     in_size=self.ac_shell_rep_size_in,
                                                     out_size=self.hDim,
                                                     dropout=self.dropout_lstm)

            self.LastBilstm = chaLink.NStepBiLSTM(n_layers=1,
                                                  in_size=self.ac_shell_rep_size_out,
                                                  out_size=self.hDim,
                                                  dropout=self.dropout_lstm)

            self.AcTypeLayer = chaLink.Linear(in_size=self.reps_for_type_classification,
                                              out_size=3,
                                              initialW=chaInit.Uniform(0.05),
                                              initial_bias=chaInit.Uniform(0.05))

            self.LinkTypeLayer = chaLink.Linear(in_size=self.reps_for_type_classification,
                                                out_size=2,
                                                initialW=chaInit.Uniform(0.05),
                                                initial_bias=chaInit.Uniform(0.05))

            self.RelationLayer = chaLink.Linear(in_size=self.span_pair_size,
                                                out_size=1,
                                                initialW=chaInit.Uniform(0.05),
                                                initial_bias=chaInit.Uniform(0.05))

            self.BowFCLayer = chaLink.Linear(in_size=self.bow_feature_size,
                                                out_size=self.bow_rep_size,
                                                initialW=chaInit.Uniform(0.05),
                                                initial_bias=chaInit.Uniform(0.05))

            self.root_embedding = chainer.Parameter(initializer=chaInit.Uniform(0.05),
                                                    shape=self.type_rep_size)

            # self.position_info[0:12]: forward position
            # self.position_info[12:24]: backward position
            # self.position_info[24:28]: paragraph type
            if self.use_elmo:
                self.elmo_task_gamma = chainer.Parameter(initializer=chaInit.Constant(1), 
                                                         shape=1)
                self.elmo_task_s = chainer.Parameter(initializer=chaInit.Constant(1), 
                                                     shape=3)

            for param in self.Bilstm.params():
                param = chaInit.Orthogonal()
            if self.lstm_ac:
                for param in self.AcBilstm.params():
                    param = chaInit.Orthogonal()
            if self.lstm_shell:
                for param in self.ShellBilstm.params():
                    param = chaInit.Orthogonal()
            for param in self.AcShellBilstm.params():
                param = chaInit.Orthogonal()
            for param in self.LastBilstm.params():
                param = chaInit.Orthogonal()

    def sequence_embed(self, embed, xs, is_param=False):
        x_len = np.array([len(x) for x in xs])
        x_section = np.cumsum(x_len[:-1])
        xs = chaFunc.concat(xs, axis=0)
        if is_param:
            ex = embed(xs)
            ex = chaFunc.dropout(ex, self.dropout_embedding)
            exs = chaFunc.split_axis(ex, x_section, 0)
        else:
            ex = embed.W.data[xs.data]
            ex = chaFunc.dropout(ex, self.dropout_embedding)
            exs = chaFunc.split_axis(ex, x_section, 0)
        return exs

    def set_embed_matrix(self, embedd_matrix):
        self.Embed_x.W.data = embedd_matrix

    def load_elmo(self, ids, xs):
        elmo_embeddings = np.concatenate([self.elmo_embed.get(str(i))[()]
                                          for i in ids], 
                                         axis=1).astype('f')
        elmo_embeddings = self.xp.asarray(elmo_embeddings)

        if self.args.elmo_layers == "weighted":
            elmo_task_s = chaFunc.softmax(self.elmo_task_s[:, None, None], 0)
            elmo_embeddings = elmo_embeddings * chaFunc.broadcast_to(elmo_task_s, 
                                                                     elmo_embeddings.shape)
            elmo_embeddings = chaFunc.sum(elmo_embeddings,
                                          axis=0)
        elif self.args.elmo_layers == "avg":
            elmo_embeddings = chaFunc.sum(elmo_embeddings,
                                          axis=0)/3
        else:
            elmo_embeddings = self.xp.asarray(elmo_embeddings)[int(self.args.elmo_layers)-1, :, :]

        if self.args.elmo_task_gamma:
            elmo_task_gamma = self.elmo_task_gamma
            elmo_embeddings = elmo_task_gamma * elmo_embeddings

        elmo_embeddings = chaFunc.dropout(elmo_embeddings,
                                          self.dropout_embedding)
        para_len = np.array([len(x) for x in xs])
        para_section = np.cumsum(para_len[:-1])
        elmo_embeddings = chaFunc.split_axis(elmo_embeddings,
                                             para_section,
                                             0)
        return elmo_embeddings

    def set_elmo(self, ELMO_PATH):
        self.elmo_embed = h5py.File(ELMO_PATH, "r")

    def position2onehot(self, inds, dim):
        inds = chaFunc.flatten(inds)
        inds = inds.data.astype('float32') % self.max_n_spans
        inds = inds.astype('int32')
        eye = self.xp.identity(dim).astype(self.xp.float32)
        onehot = chaFunc.embed_id(inds, eye)
        return onehot

    def get_span_reps(self, x_spans, ys_l, split_axis=False):
        """
        x_spans (batchsize, n_spans, 2)
        ys_l: (batchsize, length, rep)
        """
        ys_l = chaFunc.pad_sequence(ys_l)
        start_mask_forward = self.xp.zeros((ys_l.shape[0], ys_l.shape[1])).astype(self.xp.bool)
        end_mask_forward = self.xp.zeros((ys_l.shape[0], ys_l.shape[1])).astype(self.xp.bool)
        start_mask_backward = self.xp.zeros((ys_l.shape[0], ys_l.shape[1])).astype(self.xp.bool)
        end_mask_backward = self.xp.zeros((ys_l.shape[0], ys_l.shape[1])).astype(self.xp.bool)

        for row, spans in enumerate(x_spans):
            for span in spans:
                start_mask_forward[row][span[0]-1] = True
                end_mask_backward[row][span[0]] = True
                end_mask_forward[row][span[1]] = True
                start_mask_backward[row][span[1]+1] = True

        start_hidden_states_forward = ys_l[:, :, :self.hDim][start_mask_forward]
        end_hidden_states_forward = ys_l[:, :, :self.hDim][end_mask_forward]
        start_hidden_states_backward = ys_l[:, :, self.hDim:][start_mask_backward]
        end_hidden_states_backward = ys_l[:, :, self.hDim:][end_mask_backward]
        span_forward = end_hidden_states_forward - start_hidden_states_forward
        span_backward = end_hidden_states_backward - start_hidden_states_backward

        context_before_span_forward = start_hidden_states_forward
        context_after_span_backward = start_hidden_states_backward

        span_reps = chaFunc.concat([span_forward,
                                    span_backward,
                                    context_before_span_forward,
                                    context_after_span_backward])
        if split_axis:
            x_len = np.array([len(x) if len(x) else 1 for x in x_spans])
            x_section = np.cumsum(x_len[:-1])
            span_reps = chaFunc.split_axis(span_reps, x_section, 0)

        return span_reps

    ###########
    # Encoder #
    ###########
    def get_bow_reps(self, ids, xs, xs_embed, position_info,
                     x_spans, shell_spans, x_position_info):

        assert len(x_spans[0]) == len(shell_spans[0])

        x_spans = [[[shell_span[0].tolist(), x_span[1].tolist()]
                    for x_span, shell_span in zip(x_spans_in_para, shell_spans_in_para)]
               for x_spans_in_para, shell_spans_in_para in zip(x_spans, shell_spans)]

        #(all_n_spans, 1) paragraph_type
        eye = self.xp.identity(4, dtype=self.xp.float32)
        para_type = [i.tolist() - self.max_n_spans*2
                     for i in self.xp.vstack(x_position_info)[:,2]]
        para_type = self.xp.vstack([eye[i] for i in para_type])

        #(batchsize, max_n_tokens, word_vec)
        xs_embed = chaFunc.pad_sequence(xs_embed, padding=-1)

        #(batchsize, n_spans, max_n_tokens, word_vec)
        xs_embed = [chaFunc.tile(xs_embed[i], (len(spans),1,1)) 
                    for i, spans in enumerate(x_spans)]

        #(all_spans_in_batch, max_n_tokens, word_vec)
        xs_embed = chaFunc.vstack(xs_embed)

        #(batchsize, max_n_tokens, word_id)
        xs = chaFunc.pad_sequence(xs, padding=0)

        #(batchsize, n_spans, max_n_tokens, word_id)
        xs = [self.xp.tile(xs[i].data, (len(spans),1)) 
              for i, spans in enumerate(x_spans)]

        #(all_spans_in_batch, max_n_tokens, word_id)
        xs = self.xp.vstack(xs)

        #(all_spans_in_batch, max_n_tokens, word_vec)
        mask_xs_embed_bool = self.xp.zeros(xs_embed.shape).astype(self.xp.bool)

        #(all_spans_in_batch, word_vec) the length of each span
        len_spans = self.xp.zeros((xs_embed.shape[0],
                                   xs_embed.shape[2])).astype(self.xp.float32)

        #(all_spans_in_batch, (start, end))
        x_spans = np.vstack(x_spans)

        xs_ids = []
        eye = self.xp.identity(len(self.vocab), dtype=self.xp.float32)

        for i, span in enumerate(x_spans):
            mask_xs_embed_bool[i][int(span[0]):int(span[1])] = True

            xs_ids.append(self.xp.sum(eye[xs[i][int(span[0]):int(span[1])]], 0))
            len_spans[i].fill(span[1]-span[0])

        # max pooling
        mask_xs_embed = self.xp.zeros(xs_embed.shape).astype(self.xp.float32)
        mask_xs_embed.fill(-self.xp.inf)
        max_pooling_xs = chaFunc.max(chaFunc.where(
            mask_xs_embed_bool,
            xs_embed,
            mask_xs_embed
        ), 1)

        # average pooling
        mask_xs_embed = self.xp.zeros(xs_embed.shape).astype(self.xp.float32)
        avg_pooling_xs = chaFunc.sum(chaFunc.where(
            mask_xs_embed_bool,
            xs_embed,
            mask_xs_embed
        ), 1) / len_spans

        # min pooling
        mask_xs_embed = self.xp.zeros(xs_embed.shape).astype(self.xp.float32)
        mask_xs_embed.fill(self.xp.inf)
        min_pooling_xs = chaFunc.min(chaFunc.where(
            mask_xs_embed_bool,
            xs_embed,
            mask_xs_embed
        ), 1)

        #(all_n_spans, max_n_tokens, vocab_size)
        xs_ids = self.xp.vstack(xs_ids)

        #(all_n_spans, feature_vector)
        if self.use_elmo:
            # We found that pooling-based features with ELMo does not significantly contribute the performance.
            # Then, we only used discrete BoW features for span-based models with ELMo.
            bow_reps = chaFunc.concat([xs_ids])
        else:
            bow_reps = chaFunc.concat([max_pooling_xs,
                                       min_pooling_xs,
                                       avg_pooling_xs,
                                       xs_ids
                                       ])

        assert bow_reps.shape[-1] == self.bow_feature_size

        bow_reps = chaFunc.sigmoid(self.BowFCLayer(bow_reps))
        bow_reps = chaFunc.dropout(bow_reps, self.dropout)

        return bow_reps

    def hierarchical_encode(self, ids, xs, xs_embed, position_info,
                            x_spans, shell_spans, x_position_info):

        _, _, ys_l = self.Bilstm(None, None, xs_embed)

        if self.lstm_ac:
            ac_reps = self.get_span_reps(x_spans, ys_l, split_axis=True)
            _, _, ys_l_ac = self.AcBilstm(None, None, ac_reps)

            ac_reps = chaFunc.vstack(ys_l_ac)
        else:
            ac_reps = self.get_span_reps(x_spans, ys_l)

        if self.lstm_shell:
            shell_reps = self.get_span_reps(shell_spans, ys_l, split_axis=True)
            _, _, ys_l_shell = self.ShellBilstm(None, None, shell_reps)
            shell_reps = chaFunc.vstack(ys_l_shell)
        else:
            shell_reps = self.get_span_reps(shell_spans, ys_l)

        ac_shell_reps = chaFunc.concat([ac_reps, shell_reps],
                                       axis=-1)

        span_reps_bow = self.get_bow_reps(ids,
                                          xs,
                                          xs_embed,
                                          position_info,
                                          x_spans,
                                          shell_spans,
                                          x_position_info)

        ac_shell_reps = chaFunc.concat([ac_shell_reps,
                                        position_info,
                                        span_reps_bow],
                                       -1)

        assert ac_shell_reps.shape[-1] == self.ac_shell_rep_size_in

        return ac_shell_reps

    ###########################
    # for link identification #
    ###########################
    def calc_pair_score(self, span_reps_pad, relative_position_info, batchsize, max_n_spans):
        #(batchsize, max_n_spans, span_representation)
        span_reps_pad_ntimes = chaFunc.tile(span_reps_pad, (1, 1, max_n_spans))

        #(batchsize, max_n_spans, max_n_spans, span_representation)
        span_reps_matrix = chaFunc.reshape(span_reps_pad_ntimes,
                                           (batchsize,
                                            max_n_spans,
                                            max_n_spans,
                                            span_reps_pad.shape[-1]))

        #(batchsize, max_n_spans, max_n_spans, span_representation)
        span_reps_matrix_t = chaFunc.transpose(span_reps_matrix, axes=(0, 2, 1, 3))

        #(batchsize, max_n_spans, max_n_spans, pair_representation)
        pair_reps = chaFunc.concat(
            [span_reps_matrix,
             span_reps_matrix_t,
             span_reps_matrix*span_reps_matrix_t,
             relative_position_info],
            axis=-1)

        #########################
        #### add root object ####
        #########################

        #(batchsize, max_n_spans, span_rep_size)
        root_matrix = chaFunc.tile(self.root_embedding, (batchsize, self.max_n_spans, 1))

        #(batchsize, max_n_spans, pair_rep_size)
        pair_reps_with_root = chaFunc.concat([span_reps_pad,
                                              root_matrix,
                                              span_reps_pad*root_matrix,
                                              self.xp.zeros((batchsize,
                                                             self.max_n_spans,
                                                             self.relative_position_info_size))
                                              .astype(self.xp.float32)],
                                             axis=-1)

        #(batchsize, max_n_spans, max_n_spans+1, pair_rep_size)
        pair_reps = chaFunc.concat([pair_reps,
                                    chaFunc.reshape(pair_reps_with_root,
                                                    (batchsize,
                                                     self.max_n_spans,
                                                     1,
                                                     self.span_pair_size))],
                                   axis=2)

        #(batchsize, max_n_spans*max_n_spans, pair_rep_size)
        pair_reps = chaFunc.reshape(pair_reps,
                                    (batchsize*max_n_spans*(max_n_spans+1),
                                     self.span_pair_size))

        #(batsize, max_n_spans*max_n_spans) calculate relation score for each pair
        pair_scores = self.RelationLayer(pair_reps)

        #(batchsize*max_n_spans, max_n_spans)
        pair_scores = chaFunc.reshape(pair_scores,
                                      (batchsize, max_n_spans, max_n_spans+1))

        return pair_scores

    def mask_link_scores(self, pair_scores, x_spans, batchsize,
                         max_n_spans, mask_type="minus_inf"):

        #(max_n_spans, max_n_spans+1)
        mask_single = self.xp.hstack((self.xp.identity(max_n_spans),
                                      self.xp.zeros((max_n_spans, 1))))

        #switch 0 and 1
        mask_single = mask_single*-1+1

        #(batchsize*max_n_spans, max_n_spans)
        mask = self.xp.tile(mask_single, (batchsize, 1))

        #(batchsize, max_n_spans, max_n_spans)
        mask = self.xp.reshape(mask, (batchsize, max_n_spans, max_n_spans+1))

        # mask
        for i in range(batchsize):
            mask[i, :, len(x_spans[i]):-1] = 0

        mask = mask.astype(self.xp.bool)

        if mask_type == "minus_inf":
            #matrix for masking
            minus_inf = self.xp.zeros((batchsize, max_n_spans, max_n_spans+1),
                                      dtype=self.xp.float32)
            minus_inf.fill(-self.xp.inf)

            #(batchsize, max_n_spans, max_n_spans+1, 1)
            masked_pair_scores = chaFunc.where(mask, pair_scores, minus_inf)

        else:
            zero = self.xp.zeros((batchsize, max_n_spans, max_n_spans+1),
                                 dtype=self.xp.float32)

            #(batchsize, max_n_spans, max_n_spans+1, 1)
            masked_pair_scores = chaFunc.where(mask, pair_scores, zero)

        return masked_pair_scores

    ################
    # data loading #
    ################
    def load_data(self, ids):
        xs = [self.essay_info_dict[int(i)]["text"] for i in ids]

        #[all_essays, [n_spans, (start, end)]]
        x_spans = [self.essay_info_dict[int(i)]["ac_spans"]
                   for i in ids]

        x_spans = [spans if len(spans) else self.xp.array([[1, len(xs[i])-2]],
                                                          dtype=self.xp.int32)
                   for i, spans in enumerate(x_spans)]

        shell_spans = [self.essay_info_dict[int(i)]["shell_spans"]
                       for i in ids]

        shell_spans = [spans if len(spans) else self.xp.array([[1, len(xs[i])-2]],
                                                              dtype=self.xp.int32) 
                       for i, spans in enumerate(shell_spans)]

        #(batchsize, max_n_spans, (ac id in essay, ac id in paragraph, paragraph id))
        x_position_info = [self.essay_info_dict[int(i)]["ac_position_info"] for i in ids]

        return xs, x_spans, shell_spans, x_position_info


    def get_position_info(self, x_position_info):
        #(n_spans, 3)
        position_info = chaFunc.vstack(x_position_info)

        # the number of ACs in a batch
        n_spans = position_info.shape[0]

        #(n_spans*3, max_n_spans)
        position_info = self.position2onehot(position_info, self.max_n_spans)

        #(n_spans, 3*max_n_spans)
        position_info = chaFunc.reshape(position_info, (n_spans, self.position_info_size))
        return position_info

    def get_relative_position_info(self, x_position_info):
        #(batchsize, max_n_spans, 3)
        span_position_info = chaFunc.pad_sequence(x_position_info, self.max_n_spans, -1)

        #(batchsize, max_n_spans, max_n_spans*3)
        span_position_info = chaFunc.tile(span_position_info, (1, 1, self.max_n_spans))

        #(batchsize, max_n_spans, max_n_spans, 3)
        span_position_info_matrix = chaFunc.reshape(span_position_info,
                                                    (self.batchsize,
                                                     self.max_n_spans,
                                                     self.max_n_spans,
                                                     3))

        #(batchsize, max_n_spans, max_n_spans, 3)
        span_position_info_matrix_t = chaFunc.transpose(span_position_info_matrix,
                                                        axes=(0, 2, 1, 3))

        #(batchsize, max_n_spans, max_n_spans, 3) 
        # relative position information
        span_relative_position_info_matrix = span_position_info_matrix - span_position_info_matrix_t

        pair_position_matrix = span_relative_position_info_matrix

        #(batchsize*max_n_spans*max_n_spans, 1)
        relative_position = pair_position_matrix.data[:, :, :, [0]] + self.max_n_spans

        #(batchsize*max_n_spans*max_n_spans, relative_position_info_size)
        relative_position_info = self.position2onehot(relative_position,
                                                      self.relative_position_info_size)

        #(batchsize, max_n_spans, max_n_spans, relative_position_info_size)
        relative_position_info = chaFunc.reshape(relative_position_info, 
                                                 (self.batchsize,
                                                  self.max_n_spans,
                                                  self.max_n_spans,
                                                  self.relative_position_info_size))
        return relative_position_info

    def get_section(self, xs, x_spans):
        self.batchsize = len(xs)
        x_len = np.array([len(x) for x in x_spans])
        x_section = np.cumsum(x_len[:-1])
        return x_section

    def majority_voting_to_links(self, position_info):
        pair_scores = self.xp.zeros((self.batchsize,
                                     self.max_n_spans,
                                     self.max_n_spans + 1)).astype(self.xp.float32)

        for i, position_info in enumerate(position_info):
            if position_info[0][-1] == 25:
                pair_scores[i, :, 0] = 1
                pair_scores[i, 0, self.max_n_spans] = 2
            else:
                pair_scores[i, :, self.max_n_spans] = 1
        return pair_scores

    def __call__(self, ids, ts_link, ts_type, ts_link_type):
        """
        Args:
        ids: essay ids
        ts_link: gold links
        ts_type: gold ac types
        ts_link_type: gold link types

        Return:
        (all_spans, candidates, score)
        """

        #############
        # load data #
        #############
        xs, x_spans, shell_spans, x_position_info= self.load_data(ids)

        assert len(xs) == len(x_spans)
        assert len(xs) == len(shell_spans)
        assert x_spans[0][0][1] >= x_spans[0][0][0]
        assert shell_spans[0][0][1] >= shell_spans[0][0][0]
        assert len(x_position_info[0][0]) == 3

        ###################
        # load embeddings #
        ###################
        if self.use_elmo:
            xs_embed = self.load_elmo(ids, xs)
        else:
            xs_embed = self.sequence_embed(self.Embed_x, xs, False)

        x_section = self.get_section(xs, x_spans)

        position_info = self.get_position_info(x_position_info)
        relative_position_info = self.get_relative_position_info(x_position_info)

        ###########
        # encoder #
        ###########
        span_reps = self.hierarchical_encode(ids,
                                             xs,
                                             xs_embed,
                                             position_info,
                                             x_spans,
                                             shell_spans,
                                             x_position_info
                                             )


        ###########
        # decoder #
        ###########
        pair_scores, ac_types, link_types, span_reps_pad =\
            self.decoder_net(span_reps,
                             x_spans,
                             x_section,
                             position_info,
                             relative_position_info,
                             ts_link)

        if self.baseline_heuristic:
            pair_scores = self.majority_voting_to_links(position_info)

        masked_pair_scores = self.mask_link_scores(pair_scores,
                                                   x_spans,
                                                   self.batchsize,
                                                   self.max_n_spans,
                                                   mask_type="minus_inf")

        #(batchsize*max_n_spans, max_n_spans+1)
        masked_pair_scores = chaFunc.reshape(masked_pair_scores,
                                             (self.batchsize*self.max_n_spans,
                                              self.max_n_spans+1))

        return masked_pair_scores, ac_types, link_types
