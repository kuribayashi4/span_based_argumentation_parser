import math
from .decode import decode_mst
import chainer
import chainer.functions as chaFunc
from chainer import reporter


class FscoreClassifier(chainer.link.Chain):

    compute_accuracy = True

    def __init__(self,
                 predictor,
                 max_n_spans,
                 args,
                 lossfun=chaFunc.softmax_cross_entropy,
                 accfun=chaFunc.accuracy,
                 fscore_target_fun=chaFunc.classification_summary,
                 fscore_link_fun=None,
                 count_prediction=None,
                 label_link_key="ts_link",
                 label_type_key="ts_type",
                 label_link_type_key="ts_link_type",
                 class_weight=None,
                 ac_type_alpha=0,
                 link_type_alpha=0):
        if not (isinstance(label_link_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_link_key))
        if not (isinstance(label_type_key, (int, str))):
            raise TypeError('label_key must be int or str, but is %s' %
                            type(label_type_key))

        super(FscoreClassifier, self).__init__()
        self.max_n_spans = max_n_spans
        self.settings = args
        self.lossfun = lossfun
        self.accfun = accfun
        self.fscore_target_fun = fscore_target_fun
        self.fscore_link_fun = fscore_link_fun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_link_key = label_link_key
        self.label_type_key = label_type_key
        self.label_link_type_key = label_link_type_key
        self.class_weight = class_weight
        self.count_prediction = count_prediction
        self.ac_type_alpha = ac_type_alpha
        self.link_type_alpha = link_type_alpha
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, *args, **kwargs):

        if isinstance(self.label_link_key, str) and isinstance(self.label_link_key, str):
            if self.label_link_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_link_key
                raise ValueError(msg)
            if self.label_type_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_type_key
                raise ValueError(msg)
            t_link = kwargs[self.label_link_key]
            t_type = kwargs[self.label_type_key]

            # flatten and remove label -1
            t_type = t_type[t_type > -1]

            t_link_type = kwargs[self.label_link_type_key]

            # flatten and remove label -1
            t_link_type = t_link_type[t_link_type > -1]

        self.y = None
        self.loss = 0
        self.accuracy = None
        self.y = self.predictor(*args, **kwargs)

        #y_link = (batchsize*N_max_spans, 13)
        self.y_link = self.y[0]
        y_link_mst = decode_mst(self.y_link, t_link, self.max_n_spans)

        # y_type = (n_spans, 3)
        self.y_type = self.y[1]
        self.y_link_type = self.y[2]

        self.loss = 0
        self.loss_link = self.lossfun(self.y_link, t_link)
        reporter.report({'loss_link': self.loss_link}, self)
        self.loss += (1 - self.ac_type_alpha -
                      self.link_type_alpha)*self.loss_link

        assert self.ac_type_alpha + self.link_type_alpha <= 1

        if self.ac_type_alpha:
            self.loss_type = chaFunc.softmax_cross_entropy(self.y_type, t_type)
            reporter.report({'loss_ac_type': self.loss_type}, self)
            self.loss += self.ac_type_alpha*self.loss_type

        if self.link_type_alpha:
            self.loss_link_type = chaFunc.softmax_cross_entropy(self.y_link_type,
                                                                t_link_type,
                                                                ignore_label=2)
            reporter.report({'loss_link_type': self.loss_link_type}, self)
            self.loss += self.link_type_alpha*self.loss_link_type

        reporter.report({'loss': self.loss}, self)

        macro_f_scores = []

        if self.compute_accuracy:

            ###########################
            # link prediction results #
            ###########################
            self.accuracy_link = self.accfun(
                y_link_mst, chainer.cuda.to_cpu(t_link))
            reporter.report({'accuracy_link': self.accuracy_link}, self)

            if self.fscore_link_fun:
                f_binary = self.fscore_link_fun(y_link_mst,
                                                chainer.cuda.to_cpu(
                                                    t_link),
                                                self.max_n_spans)
                f_link = f_binary[0]
                f_nolink = f_binary[1]
                macro_f_link = (f_link+f_nolink)/2

                if not math.isnan(macro_f_link) and self:
                    macro_f_scores.append(macro_f_link)

                reporter.report({'f_link': f_link}, self)  # f score
                reporter.report({'f_nolink': f_nolink}, self)  # f score
                reporter.report(
                    {'macro_f_link': macro_f_link}, self)  # f score

            ##############################
            # ac_type prediction results #
            ##############################
            self.accuracy_type = chaFunc.accuracy(self.y_type, t_type)
            reporter.report({'accuracy_type': self.accuracy_type}, self)

            if self.settings.dataset == "PE":
                self.summary_type = chaFunc.classification_summary(self.y_type,
                                                                   t_type,
                                                                   label_num=3)
                f_type = self.summary_type[2]
                support_type = self.summary_type[3]
                f_premise = f_type[0]
                f_claim = f_type[1]
                f_majorclaim = f_type[2]
                macro_f_type = sum(f_type)/len(f_type)
            else:
                self.summary_type = chaFunc.classification_summary(self.y_type,
                                                                   t_type,
                                                                   label_num=2)
                f_type = self.summary_type[2]
                support_type = self.summary_type[3]
                f_premise = f_type[0]
                f_claim = f_type[1]
                f_majorclaim = 0
                macro_f_type = sum(f_type)/len(f_type)

            if self.ac_type_alpha:
                if math.isnan(macro_f_type.data):
                    macro_f_scores.append(0)
                else:
                    macro_f_scores.append(macro_f_type)

            reporter.report({'f_premise': f_premise}, self)
            reporter.report({'f_claim': f_claim}, self)
            reporter.report({'f_majorclaim': f_majorclaim}, self)
            reporter.report({'macro_f_type': sum(f_type)/len(f_type)}, self)
            reporter.report({'ac_type_predicted_class_{}'.format(i):
                             self.count_prediction(self.y_type, i)

                             for i, val in enumerate(support_type)}, self)
            reporter.report({'ac_type_gold_class_{}'.format(i): val
                             for i, val in enumerate(support_type)}, self)

            ################################
            # link type prediction results #
            ################################
            self.accuracy_type = chaFunc.accuracy(self.y_link_type,
                                                  t_link_type,
                                                  ignore_label=2)
            reporter.report({'accuracy_link_type': self.accuracy_type}, self)

            self.summary_type = chaFunc.classification_summary(self.y_link_type,
                                                               t_link_type,
                                                               label_num=2,
                                                               ignore_label=2)
            f_type = self.summary_type[2]
            support_type = self.summary_type[3]
            f_support = f_type[0]
            f_attack = f_type[1]
            macro_f_type = sum(f_type)/len(f_type)

            if self.link_type_alpha:
                if math.isnan(macro_f_type.data):
                    macro_f_scores.append(0)
                else:
                    macro_f_scores.append(macro_f_type)

            reporter.report({'f_support': f_support}, self)
            reporter.report({'f_attack': f_attack}, self)
            reporter.report(
                {'macro_f_link_type': sum(f_type)/len(f_type)}, self)
            reporter.report({'link_type_predicted_class_{}'.format(i):
                             self.count_prediction(self.y_link_type, i)

                             for i, val in enumerate(support_type)}, self)
            reporter.report({'link_type_gold_class_{}'.format(i): val
                             for i, val in enumerate(support_type)}, self)

            reporter.report({'total_macro_f': sum(
                macro_f_scores)/len(macro_f_scores)}, self)
        return self.loss
