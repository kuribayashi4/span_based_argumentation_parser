import argparse
import os
import json
import sys
import dill
import numpy as np
from collections import defaultdict
from easydict import EasyDict as edict

from classifier.decode import decode_mst
from classifier.parsing_loss import FscoreClassifier
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
from utils.option import post_process_args_info

import chainer
import chainer.functions as chaFunc
from chainer import cuda
from chainer.serializers import load_npz


def set_xp(args):
    global xp
    if args.device:
        xp = cuda.cupy
        cuda.check_cuda_available()
        cuda.get_device(args.gpu_id).use()
    else:
        xp = np


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory",
                        "-d",
                        required=True,
                        help="directory which has trained model"
                        )
    parser.add_argument("--gpu-id",
                        "-g",
                        default=-1,
                        type=int
                        )
    parser.add_argument("--device",
                        action='store_true'
                        )
    parser.add_argument("--data",
                        choices=['dev', 'test'],
                        default='test'
                        )
    return parser.parse_args()


def main():

    args_infer = parse_args()
    directory = args_infer.directory

    args = edict(json.load(open(os.path.join(directory, "config.json"), "r")))
    args.gpu_id = args_infer.gpu_id
    args.device = args_infer.device
    args = post_process_args_info(args)

    reset_seed(args)
    set_xp(args)

    ################
    # load dataset #
    ################
    sys.stderr.write("Loading essay data...\n")
    vocab = read_vocabfile(args.vocab_path)
    essay_info_dict, essay_max_n_dict, para_info_dict = \
        get_data_dicts(vocab,
                       args)

    if args.dataset == "PE":
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

    ################
    # select model #
    ################
    sys.stderr.write("Loading setting data...\n")
    predictor = dill.load(open(os.path.join(directory, "model.setting"), "rb"))

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
                link_type_alpha=args.link_type_alpha,
                )

    if args.gpu_id >= 0:
        model.to_gpu()

    sys.stderr.write("Loading parameters...\n")
    load_npz(os.path.join(directory, "model.best"),
             model)

    #####################
    # loading embedding #
    #####################
    load_embeddings(model, vocab, args)

    index2token = {v: k for k, v in vocab.items()}
    attack_attack2acc = []
    attack_attack2c = 0

    depth2acc = defaultdict(list)
    depth2c = defaultdict(int)
    
    if args_infer.data == 'test':
        infer_iter = test_iter
    elif args_infer.data == 'dev':
        infer_iter = dev_iter
    else:
        print("invalid data name")
        exit()

    print("{}: {}\t{}\t{}".format("AC_num", "AC_text", "gold", "prediction"))
    with chainer.using_config('train', False), chainer.no_backprop_mode():

        for test_sample in infer_iter.next():
            data = convert([test_sample], args.gpu_id)
            t_link = data["ts_link"]
            t_type = data["ts_type"]
            t_type = t_type[t_type > -1]
            t_link_type = data["ts_link_type"]
            t_link_type = t_link_type[t_link_type > -1]
            y = model.predictor(**data)
            y_link = y[0]

            y_link_mst = decode_mst(y_link, t_link, max_n_spans_para)

            y_class = chaFunc.argmax(y_link_mst, axis=-1)
            mask = t_link > -1
            mask = chainer.cuda.to_cpu(mask)
            t_masked = t_link[mask]
            y_masked = y_class[mask.flatten()]
            essay_info = essay_info_dict[test_sample[0]]
            ac_i = 0
            for span, shell_span, depth, gold, prediction in zip(essay_info["ac_spans"], 
                                                                 essay_info["shell_spans"], 
                                                                 essay_info["ac_relation_depth"], 
                                                                 t_masked,
                                                                 y_masked):
                text = essay_info["text"]
                shell_text = " ".join([index2token[int(token)]
                                       for token in text[int(shell_span[0]):int(shell_span[1])+1]])
                ac_text = " ".join([index2token[int(token)]
                                    for token in text[int(span[0]):int(span[1])+1]])
                print("{}: {} {}\t{}\t{}".format(ac_i, shell_text, ac_text, int(gold), int(prediction.data)))

                rel_type = essay_info["ac_relation_types"][ac_i]
                if essay_info["relation_children"][ac_i]:
                    if any([essay_info["ac_relation_types"][child].tolist() == 1 
                            for child in essay_info["relation_children"][ac_i]]):

                        attack_attack2acc.append(gold == prediction.data)
                        attack_attack2c += 1

                if rel_type == 1:
                    target = essay_info["relation_targets"][ac_i]
                    if target != max_n_spans_para:
                        if essay_info["ac_relation_types"][target] == 1:
                            attack_attack2acc.append(gold == prediction.data)
                            attack_attack2c += 1

                depth2acc[int(depth)].append(gold == prediction.data)
                depth2c[int(depth)] += 1

                ac_i += 1
    print()

    for i in range(len(depth2acc)):
        print("depth {}".format(i))
        print(depth2c[i])
        print(sum(depth2acc[i])/len(depth2acc[i]))

    print()
    print(attack_attack2c)
    print(sum(attack_attack2acc)/len(attack_attack2acc))


if __name__ == "__main__":
    main()
