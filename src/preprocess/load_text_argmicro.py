import json
import argparse
import tqdm
import xml.etree.ElementTree as ET
from collections import defaultdict
from nltk import word_tokenize
from emnlp2015.util.folds import folds
from emnlp2015.util.arggraph import ArgGraph


def distinguish_ac_am(text, am_list):
    candidate_ams = []
    for am in am_list:
        index = text.lower().find(am.lower())
        if index == 0:
            candidate_ams.append(am.lower())
            if text.lower().find(am.lower()+" ,") == 0:
                candidate_ams.append(am.lower()+" ,")
            if text.lower().find(am.lower().rstrip(",")) == 0:
                candidate_ams.append(am.lower().rstrip(","))
    if candidate_ams:
        am = sorted(candidate_ams, key=lambda x: len(x), reverse=True)[0]
        ac = " ".join(text.lower().split(am)[1:])
    else:
        am = ""
        ac = text

    if not ac[-1].isalpha():
        suf = ac[-1]
        ac = ac[:-1]
    else:
        suf = ""
    text = am + " <AC> " + ac + " </AC> " + suf
    return text


def make_author_folds(all_tids, tid2id):
    folds_50 = defaultdict(list)
    set_all_tids = set(all_tids)

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


def main(should_use_tag):
    all_tids = sorted(list(set([tid for fold in folds for tid in fold])))
    set_all_tids = set(all_tids)
    assert len(set_all_tids) == 112

    tid2id = defaultdict(lambda: len(tid2id)*2+1)
    for tid in all_tids:
        tid2id[tid]
    tid2id = dict(tid2id)

    # tid2id = json.load(open("./work/argmicro_folds/tid2id.json"))
    relation_type_dict = {"reb": "Attack",
                          "und": "Attack",
                          "sup": "Support",
                          "exa": "Support",
                          "ROOT": "None"
                          }
    am_list = json.load(open("./work/am.json"))

    essay_i = 0
    print("Essay_id\tParagraph_id\tParagraph_id_in_essay\tParagraph_type\t"
          "Paragraph_stance\tAC_id_in_essay\tAC_id_in_paragraph\tAC_type\t"
          "AC_parent\tAC_relation_type\ttoken_id_in_essay\ttoken_id_in_paragraph\ttoken")
    for tid, text_i in tqdm.tqdm(tid2id.items()):
        g = ArgGraph()
        g.load_from_xml(
            "./data/arg-microtexts-master/corpus/en/" + tid + ".xml")

        # topic information
        tree = ET.parse(
            "./data/arg-microtexts-master/corpus/en/" + tid + ".xml")
        root = tree.getroot()
        topic = root.get("topic_id")
        topic_token_i = 0
        topic_i = text_i - 1
        if topic:
            topic_tokens = topic.split("_")

            if should_use_tag:
                print("{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-\t-\t{}\t{}\t<prompt>"
                      .format(essay_i,
                              topic_i,
                              0,
                              "prompt",
                              None,
                              topic_token_i,
                              topic_token_i
                              )
                      )

                topic_token_i += 1
            for token in topic_tokens:
                print("{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-\t-\t{}\t{}\t{}"
                      .format(essay_i,
                              topic_i,
                              0,
                              "prompt",
                              None,
                              topic_token_i,
                              topic_token_i,
                              token
                              )
                      )
                topic_token_i += 1

            if should_use_tag:
                print("{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-\t-\t{}\t{}\t</prompt>"
                      .format(essay_i,
                              topic_i,
                              0,
                              "prompt",
                              None,
                              topic_token_i,
                              topic_token_i))
                topic_token_i += 1
        else:
            if should_use_tag:
                print("{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-\t-\t{}\t{}\t<prompt>"
                      .format(essay_i,
                              topic_i,
                              0,
                              "prompt",
                              None,
                              topic_token_i,
                              topic_token_i
                              )
                      )
                topic_token_i += 1

            print("{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-\t-\t{}\t{}\t{}"
                  .format(essay_i,
                          topic_i,
                          0,
                          "prompt",
                          None,
                          topic_token_i,
                          topic_token_i,
                          "@@"
                          )
                  )
            topic_token_i += 1
            if should_use_tag:
                print("{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-\t-\t{}\t{}\t</prompt>"
                      .format(essay_i,
                              topic_i,
                              0,
                              "prompt",
                              None,
                              topic_token_i,
                              topic_token_i
                              )
                      )
                topic_token_i += 1

        # text information
        relations = g.get_adus_as_dependencies()
        relations = sorted(relations, key=lambda x: x[0])
        token_i = 0
        text_i = text_i
        if should_use_tag:
            print("{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-\t-\t{}\t{}\t<para-body>"
                  .format(essay_i,
                          text_i,
                          0,
                          "body",
                          None,
                          token_i,
                          token_i
                          )
                  )
            token_i += 1
        # print("<para-body>", end=" ")

        for ac_i, relation in enumerate(relations):
            adu_id = relation[0].lstrip("a")
            relative_distance = int(relation[1].lstrip("a")) - int(adu_id)
            relation_type = relation[2]
            text = g.nodes["e"+adu_id]["text"]
            if relation_type == "add":
                target = relation[1]
                while(relation_type == "add"):
                    for rel in relations:
                        if rel[0] == target:
                            relation_type = rel[2]
                            target = rel[1]
                        if relation_type != "add":
                            break
                relative_distance = int(target.lstrip("a")) - int(adu_id)
            if relation_type == "ROOT":
                ac_type = "Claim"
                relative_distance = "None"
            else:
                ac_type = "Premise"
            text = " ".join(word_tokenize(text))
            text = distinguish_ac_am(text, am_list)

            ac_flag = 0
            for token in text.split():
                if token == "<AC>":
                    ac_flag = 1
                    if should_use_tag:
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t<AC>"
                              .format(essay_i,
                                      text_i,
                                      0,
                                      "body",
                                      None,
                                      ac_i,
                                      ac_i,
                                      ac_type,
                                      relative_distance,
                                      relation_type_dict[relation_type],
                                      token_i,
                                      token_i
                                      )
                              )
                        token_i += 1
                elif token == "</AC>":
                    ac_flag = 0
                    if should_use_tag:
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t</AC>"
                              .format(essay_i,
                                      text_i,
                                      0,
                                      "body",
                                      None,
                                      ac_i,
                                      ac_i,
                                      ac_type,
                                      relative_distance,
                                      relation_type_dict[relation_type],
                                      token_i,
                                      token_i
                                      )
                              )
                        token_i += 1
                else:
                    if ac_flag:
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                              .format(essay_i,
                                      text_i,
                                      0,
                                      "body",
                                      None,
                                      ac_i,
                                      ac_i,
                                      ac_type,
                                      relative_distance,
                                      relation_type_dict[relation_type],
                                      token_i,
                                      token_i,
                                      token
                                      )
                              )
                        token_i += 1
                    else:
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                              .format(essay_i,
                                      text_i,
                                      0,
                                      "body",
                                      None,
                                      "-",
                                      "-",
                                      "-",
                                      "-",
                                      "-",
                                      token_i,
                                      token_i,
                                      token
                                      )
                              )
                        token_i += 1

        if should_use_tag:
            print("{}\t{}\t{}\t{}\t{}\t-\t-\t-\t-\t-\t{}\t{}\t</para-body>"
                  .format(essay_i,
                          text_i,
                          0,
                          "body",
                          None,
                          token_i,
                          token_i
                          )
                  )

        essay_i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="store_true")

    args = parser.parse_args()
    main(args.tag)

