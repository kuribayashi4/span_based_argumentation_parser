import argparse
import sys
import dill
import copy
import numpy as np
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import training
from chainer.training import extensions, triggers

from nn.span_selection_model import SpanSelectionParser
from nn.pointer_net import PointerNetParser
from classifier.parsing_loss import FscoreClassifier

from utils.resource import Resource
from utils.converters import convert_hybrid as convert
from utils.loaders import (
    get_data_dicts,
    return_train_dev_test_ids_PE,
    return_train_dev_test_iter_PE, return_train_dev_test_iter_MT
)
from utils.misc import (
    reset_seed, read_vocabfile, load_glove
)
from utils.evaluators import (
    softmax_cross_entropy_flatten, accuracy_flatten,
    classification_summary_flatten, count_prediction,
    fscore_binary
)
from utils.extensions import (
    log_current_score, log_current_output
)
from utils.option import (
    add_dataset_args, add_default_args, add_embed_args,
    add_log_args, add_model_arch_args, add_optim_args,
    add_trainer_args, post_process_args_info
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    parser = add_dataset_args(parser)
    parser = add_embed_args(parser)
    parser = add_log_args(parser)
    parser = add_model_arch_args(parser)
    parser = add_optim_args(parser)
    parser = add_trainer_args(parser)

    return parser.parse_args()


def set_xp(args):
    global xp
    if args.gpu_id != -1 and args.device:
        xp = cuda.cupy
        cuda.check_cuda_available()
        cuda.get_device(args.gpu_id).use()
    else:
        xp = np


def log_info(args):
    resource = Resource(args, train=True)
    resource.dump_git_info()
    resource.dump_command_info()
    resource.dump_python_info()
    resource.dump_library_info()
    resource.save_vocab_file()
    resource.save_config_file()
    outdir = resource._return_output_dir()
    return outdir


def load_embeddings(model, vocab, args):
    if args.use_elmo:
        ################
        # ELMo setting #
        ################
        model.predictor.set_elmo(args.elmo_path)

    else:
        #################
        # glove setting #
        #################

        sys.stderr.write("Loading glove embeddings...\n")
        embedding_matrix = load_glove(args.glove_path,
                                      vocab,
                                      args.eDim)
        if args.device:
            embedding_matrix = xp.asarray(embedding_matrix)

        sys.stderr.write("Setting embed matrix...\n")
        model.predictor.set_embed_matrix(embedding_matrix)


def train(outdir, args):

    #######################
    # initialize settings #
    #######################
    vocab = read_vocabfile(args.vocab_path)
    n_epochs = args.epoch
    lr = args.lr

    #################
    # get iterators #
    #################
    if args.dataset == "PE":
        essay_info_dict, essay_max_n_dict, para_info_dict = \
            get_data_dicts(vocab, args)
        train_ids, dev_ids, test_ids = \
            return_train_dev_test_ids_PE(vocab,
                                         essay_info_dict,
                                         essay_max_n_dict,
                                         para_info_dict,
                                         args,
                                         dev_shuffle=True)

        train_iter, dev_iter, test_iter = \
            return_train_dev_test_iter_PE(train_ids,
                                          dev_ids,
                                          test_ids,
                                          essay_info_dict,
                                          essay_max_n_dict,
                                          args)

    elif args.dataset == "MT":
        train_iter, dev_iter, test_iter, essay_info_dict, \
            essay_max_n_dict, para_info_dict = \
            return_train_dev_test_iter_MT(
                                         vocab,
                                         args,
                                         args.iteration,
                                         args.fold)

    max_n_spans_para = essay_max_n_dict["max_n_spans_para"]
    max_n_paras = essay_max_n_dict["max_n_paras"]
    max_n_tokens = essay_max_n_dict["max_n_tokens"]

    ################
    # select model #
    ################
    predictor = SpanSelectionParser(
        vocab=vocab,
        essay_info_dict=essay_info_dict,
        para_info_dict=para_info_dict,
        max_n_spans_para=max_n_spans_para,
        max_n_paras=max_n_paras,
        max_n_tokens=max_n_tokens,
        settings=args,
        baseline_heuristic=args.baseline_heuristic,
        use_elmo=args.use_elmo,
        decoder=args.decoder
        )

    sys.stderr.write("dump setting file...\n")
    predictor.to_cpu()

    dill.dump(predictor,
              open(outdir + "/model.setting", "wb"))

    model = FscoreClassifier(
                predictor,
                max_n_spans_para,
                args,
                lossfun=softmax_cross_entropy_flatten,
                accfun=accuracy_flatten,
                fscore_target_fun=classification_summary_flatten,
                fscore_link_fun=fscore_binary,
                count_prediction=count_prediction,
                ac_type_alpha=args.ac_type_alpha,
                link_type_alpha=args.link_type_alpha
                )

    if args.gpu_id >= 0:
        # Specify GPU ID from command line
        model.to_gpu()

    #####################
    # loading embedding #
    #####################
    load_embeddings(model, vocab, args)

    #############
    # optimizer #
    #############
    if args.optimizer == "Adam":
        optimizer = optimizers.Adam(alpha=lr)
    else:
        raise NotImplementedError

    optimizer.setup(model)

    ##############################
    # training iteration setting #
    ##############################
    updater = training.StandardUpdater(copy.deepcopy(train_iter),
                                       optimizer,
                                       device=args.gpu_id,
                                       converter=convert)

    trainer = training.Trainer(updater,
                               (n_epochs, 'epoch'),
                               out=outdir)

    ##############
    # extensions #
    ##############
    chainer.training.triggers.IntervalTrigger(period=1,
                                              unit='epoch')

    trainer.extend(extensions.Evaluator(copy.deepcopy(dev_iter),
                                        model,
                                        converter=convert,
                                        device=args.gpu_id),
                   name='validation')
    trainer.extend(extensions.Evaluator(copy.deepcopy(test_iter),
                                        model,
                                        converter=convert,
                                        device=args.gpu_id),
                   name='test')
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
    trainer.extend(extensions.PrintReport(
        ['epoch',
         'main/loss',
         'validation/main/macro_f_link',
         'validation/main/macro_f_link_type',
         'validation/main/macro_f_type']))
         #'elapsed_time']))

    trainer.extend(log_current_score(trigger=triggers.MaxValueTrigger(
        key=args.monitor,
        trigger=(1, 'epoch')),

                                     log_report='LogReport',
                                     out_dir=outdir))

    if args.log_output:
        trainer.extend(log_current_output(trigger=triggers.MaxValueTrigger(
            key=args.monitor,
            trigger=(1, 'epoch')),
                                          test_iter=copy.deepcopy(test_iter).next(),
                                          max_n_spans_para=max_n_spans_para,
                                          log_report='LogReport',
                                          out=outdir,
                                          settings=args))

    trainer.extend(extensions.snapshot_object(model, 'model.best'),
                   trigger=chainer.training.triggers.MaxValueTrigger(
                       key=args.monitor,
                       trigger=(1, 'epoch')))

    trainer.run()


def main():

    args = parse_args()
    args = post_process_args_info(args)

    reset_seed(args)
    set_xp(args)
    outdir = log_info(args)

    train(outdir, args)


if __name__ == "__main__":
    main()
