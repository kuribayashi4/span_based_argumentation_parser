import chainer
import chainer.functions as chaFunc
from chainer import training
from utils.evaluators import fscore_binary
from utils.converters import convert_hybrid as convert
from classifier.decode import decode_mst
from sklearn.metrics import f1_score


def log_current_score(trigger, log_report, out_dir):
    global log
    global out
    global data

    log = log_report
    out = out_dir

    @training.make_extension(trigger=trigger)
    def _log_current_score(trainer):
        logs = trainer.get_extension(log)
        data = "test"
        current_result = "best score: epoch {}\nmacro_f_link: {}\t"\
            "f_link: {}\tf_nolink: {}\nmacro_f_type: {}\tf_premise: {}\t"\
            "f_claim: {}\tf_major_claim: {}\nf_macro_link_type: {}\t"\
            "f_support: {}\tf_attack: {}".format(
                logs.log[-1]['epoch'],
                round(logs.log[-1]['{}/main/macro_f_link'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_link'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_nolink'.format(data)], 3),
                round(logs.log[-1]['{}/main/macro_f_type'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_premise'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_claim'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_majorclaim'.format(data)], 3),
                round(logs.log[-1]
                      ['{}/main/macro_f_link_type'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_support'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_attack'.format(data)], 3),
            )
        with open(out_dir + "/result.txt", "a") as f:
            f.write(current_result)
            f.write("\n")
            f.write("\n")

        data = "validation"
        current_result = "best score: epoch {}\nmacro_f_link: {}\t"\
            "f_link: {}\tf_nolink: {}\nmacro_f_type: {}\tf_premise: {}\t"\
            "f_claim: {}\tf_major_claim: {}\nf_macro_link_type: {}\t"\
            "f_support: {}\tf_attack: {}".format(
                logs.log[-1]['epoch'],
                round(logs.log[-1]['{}/main/macro_f_link'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_link'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_nolink'.format(data)], 3),
                round(logs.log[-1]['{}/main/macro_f_type'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_premise'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_claim'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_majorclaim'.format(data)], 3),
                round(logs.log[-1]
                      ['{}/main/macro_f_link_type'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_support'.format(data)], 3),
                round(logs.log[-1]['{}/main/f_attack'.format(data)], 3),
            )
        with open(out_dir + "/result_dev.txt", "a") as f:
            f.write(current_result)
            f.write("\n")
            f.write("\n")
    return _log_current_score


def log_current_output(trigger, test_iter, max_n_spans_para, log_report, out, settings):
    global log
    global out_dir
    global iteration
    global n_spans
    global args

    log = log_report
    out_dir = out
    iteration = test_iter
    n_spans = max_n_spans_para
    args = settings

    @training.make_extension(trigger=trigger)
    def _log_current_output(trainer):
        logs = trainer.get_extension(log)
        model = trainer.updater.get_optimizer('main').target

        with open(out_dir + "/out_link_epoch_{}.txt".format(
            logs.log[-1]['epoch']), "w") as f1, \
                open(out_dir + "/out_ac_type_epoch_{}.txt".format(
                    logs.log[-1]['epoch']), "w") as f2, \
                open(out_dir + "/out_link_type_epoch_{}.txt".format(
                    logs.log[-1]['epoch']), "w") as f3, \
                open(out_dir + "/out_link.txt", "w") as f4, \
                open(out_dir + "/out_ac_type.txt", "w") as f5, \
                open(out_dir + "/out_link_type.txt", "w") as f6:
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                for test_sample in iteration:
                    data = convert([test_sample], args.gpu_id)
                    t_link = data["ts_link"]
                    t_type = data["ts_type"]
                    t_type = t_type[t_type > -1]
                    t_link_type = data["ts_link_type"]
                    t_link_type = t_link_type[t_link_type > -1]
                    y = model.predictor(**data)
                    y_link = y[0]
                    y_ac_type = y[1]
                    y_link_type = y[2]

                    y_link_mst = decode_mst(y_link,
                                            t_link,
                                            max_n_spans_para)

                    f_binary = fscore_binary(y_link_mst,
                                             chainer.cuda.to_cpu(t_link),
                                             n_spans)

                    f_link = f_binary[0]
                    f_nolink = f_binary[1]
                    macro_f_link = (f_link+f_nolink)/2
                    f1.write(str(macro_f_link))
                    f4.write(str(macro_f_link))
                    f1.write("\n")
                    f4.write("\n")

                    macro_f_ac_type = f1_score(t_type.tolist(),
                                               chaFunc.argmax(y_ac_type,
                                                              axis=1).data.tolist(),
                                               average="macro")
                    f2.write(str(macro_f_ac_type))
                    f5.write(str(macro_f_ac_type))
                    f2.write("\n")
                    f5.write("\n")

                    macro_f_link_type = f1_score(t_link_type.tolist(),
                                                 chaFunc.argmax(y_link_type,
                                                                axis=1).data.tolist(),
                                                 average="macro")
                    f3.write(str(macro_f_link_type))
                    f6.write(str(macro_f_link_type))
                    f3.write("\n")
                    f6.write("\n")

    return _log_current_output
