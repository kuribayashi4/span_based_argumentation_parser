import chainer.functions as chaFunc
from nn.base_model import BaseArgStrParser


class SpanSelectionParser(BaseArgStrParser):
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

        super().__init__(
            vocab,
            essay_info_dict,
            para_info_dict,
            max_n_spans_para,
            max_n_paras,
            max_n_tokens,
            settings,
            baseline_heuristic,
            use_elmo,
            decoder)

        self.decoder_net = self.span_selection_model

    def span_selection_model(self, span_reps, x_spans, x_section,
                             position_info, relative_position_info, ts_link):
        ######################
        # AC type prediction #
        ######################

        if self.lstm_ac_shell:
            span_reps = chaFunc.split_axis(span_reps, x_section, 0)
            _, _, ys_l_ac_shell = self.AcShellBilstm(None, None, span_reps)
            span_reps = chaFunc.vstack(ys_l_ac_shell)

        span_reps = chaFunc.dropout(span_reps, self.dropout)

        ac_types = self.AcTypeLayer(span_reps)
        link_types = self.LinkTypeLayer(span_reps)

        if self.lstm_type:
            last_reps = chaFunc.split_axis(span_reps, x_section, 0)
            _, _, ys_l_last = self.LastBilstm(None, None, last_reps)
            span_reps = chaFunc.vstack(ys_l_last)

        span_reps = chaFunc.dropout(span_reps, ratio=self.dropout)

        ############################
        # span pair representation #
        ############################

        # (batchsize, n_span, span_representation)
        span_reps = chaFunc.split_axis(span_reps, x_section, 0)

        # (batchsize, max_n_spans, span_representation)
        span_reps_pad = chaFunc.pad_sequence(span_reps, self.max_n_spans, -1)

        pair_scores = self.calc_pair_score(span_reps_pad,
                                           relative_position_info,
                                           self.batchsize,
                                           self.max_n_spans)

        return pair_scores, ac_types, link_types, span_reps_pad
