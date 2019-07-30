import json
from collections import defaultdict
from emnlp2015.util.folds import folds


def make_author_folds():
    all_tids = sorted(list(set([tid for fold in folds for tid in fold])))
    folds_50 = defaultdict(list)
    tid2id = defaultdict(lambda: len(tid2id)*2+1)
    for tid in all_tids:
        tid2id[tid]
    tid2id = dict(tid2id)
    set_all_tids = set(all_tids)
    assert len(set_all_tids) == 112

    for i_fold, test_tids in enumerate(folds):
        i_iteration = i_fold // 5
        set_test_tids = set(test_tids)
        assert i_iteration < 10
        assert len(set_test_tids) < 26

        train_tids = set_all_tids - set_test_tids
        train_ids = list([tid2id[tid] for tid in train_tids])
        test_ids = list([tid2id[tid] for tid in test_tids])
        folds_50[i_iteration].append([train_ids, test_ids])
    folds_50 = dict(folds_50)
    print(json.dumps(folds_50))


if __name__ == "__main__":
    make_author_folds()
