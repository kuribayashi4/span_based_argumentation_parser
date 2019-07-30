import random
import os
from collections import defaultdict
import itertools
import sys
import json
import numpy as np
import numpy as xp

from chainer import cuda
from chainer import iterators

from .misc import count_relations


def read_vocabfile(VOCABFILE):
    token2index = defaultdict(lambda: len(token2index))
    token2index["<unk>"]
    with open(VOCABFILE) as f:
        for line in f:
            token, *_ = line.split("\t")
            token2index[token]
    return token2index


def relation_info2relation_matrix(ac_relations, max_n_spans, n_spans, settings):

    """Summary line.

    Args:
        arg1 (list): list of argumentative information tuple (source, target, relation_type)

    Returns:
        ndarray: flatten version of relation matrix
                                 (axis0: source_AC_id,
                                  axis1: target_AC_id,
                                  value: relation type)
    """

    relation_matrix = xp.zeros((max_n_spans, max_n_spans)).astype('int32')
    relation_matrix.fill(-1)
    relation_matrix[n_spans:, :] = -1
    relation_matrix[:, n_spans:] = -1
    for combination in ac_relations:
        source_i = combination[0]
        target_i = combination[1]
        relation_type = combination[2]
        relation_matrix[source_i, target_i] = relation_type

    return relation_matrix.flatten().astype('int32')


def relation_info2target_sequence(ac_relations, ac_types, max_n_spans, n_spans, settings):

    """Summary line.

    Args:
        arg1 (list): list of argumentative information tuple (source, target, relation_type)

    Returns:
        array: array of target ac index
    """

    relation_seq = xp.zeros(max_n_spans).astype('int32')
    relation_type_seq = xp.zeros(max_n_spans).astype('int32')
    direction_seq = xp.zeros(max_n_spans).astype('int32')
    depth_seq = xp.zeros(max_n_spans).astype('int32')

    relation_seq.fill(max_n_spans)
    relation_type_seq.fill(2)
    direction_seq.fill(2)
    depth_seq.fill(100)
    relation_seq[n_spans:] = -1
    relation_type_seq[n_spans:] = -1
    direction_seq[n_spans:] = -1
    depth_seq[n_spans:] = -1

    for combination in ac_relations:
        source_i = combination[0]
        target_i = combination[1]
        relation_seq[source_i] = target_i
        relation_type_seq[source_i] = combination[2]

    for i in range(len(relation_seq)):
        depth = 0
        target_i = relation_seq[int(i)]
        if target_i == -1:
            continue
        while(1):
            if target_i == max_n_spans:
                break
            else:
                target_i = relation_seq[int(target_i)]
                depth += 1
        depth_seq[i] = depth

    return relation_seq, relation_type_seq, depth_seq


def relation_info2children_sequence(ac_relations, max_n_spans, n_spans, settings):
    children_list = [[] for _ in range(max_n_spans + 1)]

    for combination in ac_relations:
        source_i = combination[0]
        target_i = combination[1]
        children_list[target_i].append(source_i)
    return children_list


def get_shell_lang_span(start, text, vocab, previous_span_end):

    EOS_tokens_list = [".",
                       "!",
                       "?",
                       "</AC>",
                       "</para-intro>",
                       "</para-body>",
                       "</para-conclusion>",
                       "</essay>"]

    EOS_ids_set = set([vocab[token.lower()]
                       for token in EOS_tokens_list if token.lower() in vocab])
    shell_lang = []
    if start == 0:
        shell_span = (start, start)
        return shell_span

    for i in range(start-1, previous_span_end, -1):
        if text[int(i)] not in EOS_ids_set:
            shell_lang.append(int(i))
        else:
            break
    if shell_lang:
        shell_start = min(shell_lang)
        shell_end = max(shell_lang)
        shell_span = (shell_start, shell_end)
    else:
        shell_span = (start-1, start-1)
    return shell_span


def get_essay_detail(essay_lines, max_n_spans, vocab, settings):

    global xp
    if settings.device > 0:
        xp = cuda.cupy
    else:
        xp = np

    essay_id = int(essay_lines[0].split("\t")[0])

    # list of (start_idx, end_idx) of each ac span
    ac_spans = []

    # text of each span
    ac_texts = []

    # type of each span (premise, claim, majorclaim)
    ac_types = []

    # in which type of paragraph each ac is (opening, body, ending)
    ac_paratypes = []

    # id of the paragraph where the ac appears
    ac_paras = []

    # id of each ac (in paragraoh)
    ac_positions_in_para = []

    # linked acs (source_ac, target_ac, relation_type)
    ac_relations = []

    # list of (startr_idx, end_idx) of each am span
    shell_spans = []

    relation2id = {"Support": 0, "Attack": 1}
    actype2id = {"Premise": 0, "Claim": 1, "Claim:For": 1, "Claim:Against": 1, "MajorClaim": 2}
    paratype2id = {"intro": 0, "body": 1, "conclusion": 2, "prompt": 3}

    relation_type_seq = xp.zeros(max_n_spans).astype('int32')
    relation_type_seq.fill(2)

    text = [vocab[line.strip().split("\t")[-1].lower()]
            if line.strip().split("\t")[-1].lower() in vocab
            else vocab["<UNK>".lower()]
            for line in essay_lines]

    previous_span_end = 0
    for ac_type, lines in itertools.groupby(essay_lines, key=lambda x: x.split("\t")[6]):
        ac_lines = list(lines)

        if ac_lines[0].split("\t")[7] != "-":
            ac_text = [ac_line.split("\t")[-1].strip() for ac_line in ac_lines]
            ac_texts.append(ac_text)

            para_i = int(ac_lines[0].split("\t")[2])
            para_type = ac_lines[0].split("\t")[3]
            ac_i = int(ac_lines[0].split("\t")[6])
            ac_type = ac_lines[0].split("\t")[7]
            start = int(ac_lines[0].split("\t")[11])
            end = int(ac_lines[-1].split("\t")[11])

            ac_positions_in_para.append(ac_i)
            ac_types.append(actype2id[ac_type])
            ac_paratypes.append(paratype2id[para_type])
            ac_paras.append(para_i)

            ac_span = (start, end)
            ac_spans.append(ac_span)

            shell_span = get_shell_lang_span(start, text, vocab, previous_span_end)
            shell_spans.append(shell_span)

            if ac_type == "Claim:For":
                relation_type_seq[ac_i] = 0
            elif ac_type == "Claim:Against":
                relation_type_seq[ac_i] = 1

            if "Claim" not in ac_lines[0].split("\t")[7]:
                ac_relations.append(
                   (ac_i,
                    ac_i + int(ac_lines[0].split("\t")[8]),
                    relation2id[ac_lines[0].split("\t")[9].strip()]))
                relation_type_seq[ac_i] = relation2id[ac_lines[0].split("\t")[9].strip()]
            previous_span_end = end

    assert len(ac_spans) == len(ac_positions_in_para)
    assert len(ac_spans) == len(ac_types)
    assert len(ac_spans) == len(ac_paratypes)
    assert len(ac_spans) == len(ac_paras)
    assert len(ac_spans) == len(shell_spans)
    assert len(relation_type_seq) == max_n_spans

    assert max(relation_type_seq).tolist() <= 2
    assert len(ac_spans) >= len(ac_relations)

    n_acs = len(ac_spans)
    relation_type_seq[n_acs:] = -1

    relation_matrix = relation_info2relation_matrix(ac_relations,
                                                    max_n_spans,
                                                    n_acs,
                                                    settings)

    assert len(relation_matrix) == max_n_spans*max_n_spans

    relation_targets, _, relation_depth = \
        relation_info2target_sequence(ac_relations,
                                      ac_types,
                                      max_n_spans,
                                      n_acs,
                                      settings)

    assert len(relation_targets) == max_n_spans
    assert len(relation_depth) == max_n_spans

    relation_children = relation_info2children_sequence(ac_relations,
                                                        max_n_spans,
                                                        n_acs,
                                                        settings)

    ac_position_info = xp.array([
                       ac_positions_in_para,
                       [(i_ac - max(ac_positions_in_para))*(-1)+max_n_spans
                        for i_ac in ac_positions_in_para],
                       [i+2*max_n_spans for i in ac_paratypes]], dtype=xp.int32).T

    assert ac_position_info.shape == (n_acs, 3)

    if not len(ac_position_info):
        para_type = ac_lines[0].split("\t")[3]
        ac_position_info = xp.array([[0,
                                      0 + max_n_spans, paratype2id[para_type] + max_n_spans*2]], 
                                    dtype=xp.int32)

    essay_detail_dict = {}
    essay_detail_dict["essay_id"] = essay_id
    essay_detail_dict["text"] = xp.array(text, dtype=xp.int32)
    essay_detail_dict["ac_spans"] = xp.array(ac_spans, dtype=xp.int32)
    essay_detail_dict["shell_spans"] = xp.array(shell_spans, dtype=xp.int32)
    essay_detail_dict["ac_types"] = xp.pad(ac_types,
                                           [0, max_n_spans-len(ac_types)],
                                           'constant',
                                           constant_values=(-1, -1))
    essay_detail_dict["ac_paratypes"] = ac_paratypes
    essay_detail_dict["ac_paras"] = ac_paras
    essay_detail_dict["ac_position_info"] = ac_position_info
    essay_detail_dict["relation_matrix"] = relation_matrix
    essay_detail_dict["relation_targets"] = relation_targets
    essay_detail_dict["relation_children"] = relation_children
    essay_detail_dict["ac_relation_types"] = relation_type_seq
    essay_detail_dict["ac_relation_depth"] = relation_depth

    return essay_detail_dict


def get_essay_info_dict(FILENAME, vocab, settings):

    global xp
    if settings.device > 0:
        xp = cuda.cupy
    else:
        xp = np

    with open(FILENAME) as f:
        n_span_para = []
        n_span_essay = []
        n_para = []
        span_index_column = 6
        for line in f:
            if line.split("\t")[span_index_column] != "-" \
                    and line.split("\t")[5] != "AC_id_in_essay" \
                    and line.split("\t")[6] != "AC_id_in_paragraph":
                n_span_essay.append(int(line.split("\t")[5]))
                n_span_para.append(int(line.split("\t")[6]))
                n_para.append(int(line.split("\t")[2]))

    max_n_spans = max(n_span_para) + 1
    max_n_paras = max(n_para) + 1

    essay_info_dict = {}
    split_column = 1

    essay2parainfo = defaultdict(dict)
    essay2paraids = defaultdict(list)
    para2essayid = dict()

    with open(FILENAME) as f:
        for essay_id, lines in itertools.groupby(f, key=lambda x: x.split("\t")[split_column]):

            if essay_id == "Essay_id" or essay_id == "Paragraph_id" or essay_id == "-":
                continue

            essay_lines = list(lines)
            para_type = essay_lines[0].split("\t")[3]
            essay_id = int(essay_lines[0].split("\t")[0])
            para_id = int(essay_lines[0].split("\t")[1])

            para2essayid[para_id] = essay_id
            essay2paraids[essay_id].append(para_id)
            essay2parainfo[essay_id][para_type] = para_id

            essay_info_dict[int(para_id)] = get_essay_detail(essay_lines,
                                                             max_n_spans,
                                                             vocab,
                                                             settings)

    max_n_tokens = max([len(essay_info_dict[essay_id]["text"])
                        for essay_id in range(len(essay_info_dict))])

    essay_max_n_dict = {}
    essay_max_n_dict["max_n_spans_para"] = max_n_spans
    essay_max_n_dict["max_n_paras"] = max_n_paras
    essay_max_n_dict["max_n_tokens"] = max_n_tokens

    para_info_dict = defaultdict(dict)
    for para_id, essay_id in para2essayid.items():
        para_info_dict[para_id]["prompt"] = essay2parainfo[essay_id]["prompt"]
        if settings.dataset == "PE":
            para_info_dict[para_id]["intro"] = essay2parainfo[essay_id]["intro"]
            para_info_dict[para_id]["conclusion"] = essay2parainfo[essay_id]["conclusion"]
            para_info_dict[para_id]["context"] = essay2paraids[essay_id]

    return essay_info_dict, essay_max_n_dict, para_info_dict


def get_data_dicts(vocab, args):

    SCRIPT_PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(SCRIPT_PATH,
                             "../../work/PE_data.tsv")

    essay_info_dict, essay_max_n_dict, para_info_dict = get_essay_info_dict(DATA_PATH,
                                                                            vocab,
                                                                            args)

    return essay_info_dict, essay_max_n_dict, para_info_dict


def return_train_dev_test_ids_PE(vocab, essay_info_dict, essay_max_n_dict,
                                 para_info_dict, args, dev_shuffle=True):

    SCRIPT_PATH = os.path.dirname(__file__)
    max_n_spans_para = essay_max_n_dict["max_n_spans_para"]
    max_n_paras = essay_max_n_dict["max_n_paras"]
    max_n_tokens = essay_max_n_dict["max_n_tokens"]

    invalid_para_inds = set([para_i for para_i, info in essay_info_dict.items()
                             if isinstance(info, dict) and len(info["ac_spans"]) < 1])

    sys.stderr.write("max_n_spans_para: {}\tmax_n_paras: {}\tmax_n_tokens: {}\t"\
                     .format(max_n_spans_para,
                             max_n_paras,
                             max_n_tokens))
    sys.stderr.write("n_vocab: {}\n".format(len(vocab)))

    train_inds = list(set(json.load(open(os.path.join(SCRIPT_PATH,
                                                      "../../work/train_paragraph_index.json")))) -\
                      invalid_para_inds)
    test_inds = list(set(json.load(open(os.path.join(SCRIPT_PATH,
                                                     "../../work/test_paragraph_index.json")))) - \
                     invalid_para_inds)
    dev_inds = list(set(json.load(open(os.path.join(SCRIPT_PATH,
                                                    "../../work/dev_paragraph_index.json")))) - \
                    invalid_para_inds)

    n_trains = len(train_inds)

    if dev_shuffle:
        all_train_inds = train_inds + dev_inds
        random.seed(args.seed)
        random.shuffle(all_train_inds)
        train_inds = all_train_inds[:n_trains]
        dev_inds = all_train_inds[n_trains:]

    assert set(train_inds) & set(test_inds) & set(dev_inds) == set()

    return train_inds, dev_inds, test_inds


def return_train_dev_test_iter_PE(train_inds, dev_inds, test_inds,
                                  essay_info_dict, essay_max_n_dict, args):

    max_n_spans_para = essay_max_n_dict["max_n_spans_para"]

    train_data = load_data(train_inds, essay_info_dict, max_n_spans_para, args)
    test_data = load_data(test_inds, essay_info_dict, max_n_spans_para, args)
    dev_data = load_data(dev_inds, essay_info_dict, max_n_spans_para, args)

    train_n_links, train_n_no_links = count_relations(train_data, max_n_spans_para)
    dev_n_links, dev_n_no_links = count_relations(dev_data, max_n_spans_para)
    test_n_links, test_n_no_links = count_relations(test_data, max_n_spans_para)

    assert train_n_links + dev_n_links == 3023
    assert train_n_no_links + dev_n_no_links == 14227
    assert test_n_links == 809
    assert test_n_no_links == 4113

    train_iter = iterators.SerialIterator(train_data,
                                          args.batchsize)
    test_iter = iterators.SerialIterator(test_data,
                                         len(test_data),
                                         repeat=False,
                                         shuffle=False)
    dev_iter = iterators.SerialIterator(dev_data,
                                        len(dev_data),
                                        repeat=False,
                                        shuffle=False)

    return train_iter, dev_iter, test_iter


def return_train_dev_test_iter_MT(vocab, args,
                                  iteration_i=0, fold_i=0):

    SCRIPT_PATH = os.path.dirname(__file__)

    # change
    DATA_PATH = os.path.join(SCRIPT_PATH,
                             "../../work/MT_data.tsv")
    essay_info_dict, essay_max_n_dict, para_info_dict = get_essay_info_dict(DATA_PATH,
                                                                            vocab,
                                                                            args)

    max_n_spans_para = essay_max_n_dict["max_n_spans_para"]
    max_n_paras = essay_max_n_dict["max_n_paras"]
    max_n_tokens = essay_max_n_dict["max_n_tokens"]

    sys.stderr.write("max_n_spans_para: {}\tmax_n_paras: {}\tmax_n_tokens: {}\t"\
                     .format(max_n_spans_para,
                             max_n_paras,
                             max_n_tokens))
    sys.stderr.write("n_vocab: {}\n".format(len(vocab)))

    fold_tids = json.load(open(os.path.join(SCRIPT_PATH,
                                            "../../work/folds_author.json")))
    all_train_inds = fold_tids[str(iteration_i)][fold_i][0]
    n_trains = len(all_train_inds)

    train_inds = all_train_inds[:int(n_trains*0.9)]
    dev_inds = all_train_inds[int(n_trains*0.9):]
    test_inds = fold_tids[str(iteration_i)][fold_i][1]

    sys.stderr.write("train_inds: {}\n".format(",".join([str(index)
                                                         for index in train_inds])))
    sys.stderr.write("total_train_essays: {}\n".format(len(train_inds)))

    sys.stderr.write("test inds: {}\n".format(",".join([str(index)
                                                        for index in test_inds])))
    sys.stderr.write("total_test_essays: {}\n".format(len(test_inds)))

    sys.stderr.write("dev inds: {}\n".format(",".join([str(index)
                                                       for index in dev_inds])))
    sys.stderr.write("total_dev_essays: {}\n\n".format(len(dev_inds)))

    assert set(train_inds) & set(test_inds) & set(dev_inds) == set()
    assert len(train_inds) + len(test_inds) + len(dev_inds) == 112

    train_data = load_data(train_inds, essay_info_dict, max_n_spans_para, args)
    test_data = load_data(test_inds, essay_info_dict, max_n_spans_para, args)
    dev_data = load_data(dev_inds, essay_info_dict, max_n_spans_para, args)

    train_n_links, train_n_no_links = count_relations(train_data, max_n_spans_para)
    dev_n_links, dev_n_no_links = count_relations(dev_data, max_n_spans_para)
    test_n_links, test_n_no_links = count_relations(test_data, max_n_spans_para)

    print("n_links: ", str(train_n_links+dev_n_links+test_n_links))
    assert train_n_links + dev_n_links + test_n_links == 464

    train_iter = iterators.SerialIterator(train_data,
                                          args.batchsize)
    test_iter = iterators.SerialIterator(test_data,
                                         len(test_data),
                                         repeat=False,
                                         shuffle=False)
    dev_iter = iterators.SerialIterator(dev_data,
                                        len(dev_data),
                                        repeat=False,
                                        shuffle=False)

    return train_iter, dev_iter, test_iter, essay_info_dict, essay_max_n_dict, para_info_dict


def load_data(essay_ids, essay_info_dict, max_n_spans, args):

    global xp
    if args.device > 0:
        xp = cuda.cupy
    else:
        xp = np

    ts_link = xp.array([essay_info_dict[int(i)]["relation_targets"]
                        for i in list(essay_ids)], dtype=xp.int32)
    ts_type = xp.array([essay_info_dict[int(i)]["ac_types"]
                        for i in list(essay_ids)], dtype=xp.int32)
    ts_link_type = xp.array([essay_info_dict[int(i)]["ac_relation_types"]
                             for i in list(essay_ids)], dtype=xp.int32)

    return list(zip(essay_ids,
                    ts_link,
                    ts_type,
                    ts_link_type))
