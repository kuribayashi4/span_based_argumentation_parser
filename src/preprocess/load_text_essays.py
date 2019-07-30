import itertools
from collections import namedtuple
import copy
import argparse
import tqdm


def main(should_use_tag):

    # loading paragraph-level information
    Paragraph = namedtuple('Paragraph', ['text', 'ACs', 'stance', 'length'])
    AC = namedtuple('AC', ['text', 'tag', 'start', 'end'])
    paragraphs = []

    with open("./data/acl2017-neural_end2end_am/data/conll/Paragraph_Level/all.dat") as f:
        for is_line, lines in itertools.groupby(f, key=lambda x: x != "\n"):
            if is_line:
                ACs = []
                stance = None

                paragraph = list(lines)
                paragraph_length = len([token.split("\t")[1]
                                        for token in paragraph])
                paragraph_text = " ".join(
                    [token.split("\t")[1] for token in paragraph])

                for is_AC, tokens in itertools.groupby(paragraph,
                                                       key=lambda x: not x.split(
                                                           "\t")[4].startswith("O")
                                                       ):
                    tokens = list(tokens)
                    indices = [int(token.split("\t")[0]) -
                               1 for token in tokens]
                    text = " ".join([token.split("\t")[1] for token in tokens])
                    tag = list(
                        set([token.split("\t")[4].lstrip("BI-") for token in tokens]))
                    assert len(tag) == 1

                    tag = tag[0].strip()
                    if "Claim:" in tag:
                        stance_cur = tag.split(":")[1]
                        if stance and stance != stance_cur:
                            stance_cur = "Mixed"
                        stance = stance_cur
                    elif tag == "MajorClaim":
                        stance_cur = "For"
                        if stance and stance != stance_cur:
                            stance_cur = "Mixed"
                        stance = stance_cur

                    ac = AC(text, tag, indices[0], indices[-1])
                    ACs.append(ac)

                paragraph = Paragraph(paragraph_text,
                                      ACs,
                                      stance,
                                      paragraph_length
                                      )

                paragraphs.append(paragraph)

    # loading essay-level information
    Essay = namedtuple('Essay', ['text', 'length'])
    essays = []

    with open("./data/acl2017-neural_end2end_am/data/conll/Essay_Level/all.dat") as f:
        for is_line, lines in itertools.groupby(f, key=lambda x: x != "\n"):
            if is_line:
                lines = list(lines)
                text_length = len([line.split("\t")[1] for line in lines])
                text = " ".join([line.split("\t")[1] for line in lines])
                essay = Essay(text, text_length)
                essays.append(essay)

    # combine paragraph-level information and essay-level information
    ParaEssay = namedtuple('paraEssay', ['text', 'paragraphs'])
    para_essays = []
    essay_i = 0
    essay_length = 0
    paragraph_list = []

    for paragraph_i, paragraph in enumerate(paragraphs):
        essay_length += paragraph.length
        paragraph_list.append(paragraph)

        if essays[essay_i].length == essay_length or essays[essay_i].length == essay_length + 1:
            para_essay = ParaEssay(
                essays[essay_i].text, copy.deepcopy(paragraph_list))
            para_essays.append(para_essay)
            essay_i += 1
            essay_length = 0
            paragraph_list = []

    # output
    print("Essay_id\tParagraph_id\tParagraph_id_in_essay\tParagraph_type\t"
          "Paragraph_stance\tAC_id_in_essay\tAC_id_in_paragraph\tAC_type\t"
          "AC_parent\tAC_relation_type\ttoken_id_in_essay\ttoken_id_in_paragraph\ttoken")
    para_id = 0

    for essay_id, para_essay in tqdm.tqdm(enumerate(para_essays)):
        essay_token_id = 0
        essay_ac_id = 0
        para_n = len(para_essay.paragraphs)

        if should_use_tag:
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t<essay>"
                  .format(essay_id,
                          "-",
                          "-",
                          "-",
                          "-",
                          "-",
                          "-",
                          "-",
                          "-",
                          "-",
                          essay_token_id,
                          "-"
                          )
                  )

            essay_token_id += 1

        for essay_para_id, paragraph in enumerate(para_essay.paragraphs):
            para_token_id = 0
            para_ac_id = 0
            para_stance = paragraph.stance

            if essay_para_id == 0:
                para_type = "prompt"
                if should_use_tag:
                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t<prompt>"
                          .format(essay_id,
                                  para_id,
                                  essay_para_id,
                                  para_type,
                                  para_stance,
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  essay_token_id,
                                  para_token_id
                                  )
                          )

                    essay_token_id += 1
                    para_token_id += 1

            elif essay_para_id == 1:
                para_type = "intro"
                if should_use_tag:
                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t<para-intro>"
                          .format(essay_id,
                                  para_id,
                                  essay_para_id,
                                  para_type,
                                  para_stance,
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  essay_token_id,
                                  para_token_id
                                  )
                          )

                    essay_token_id += 1
                    para_token_id += 1

            elif essay_para_id == para_n - 1:
                para_type = "conclusion"
                if should_use_tag:
                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t<para-conclusion>"
                          .format(essay_id,
                                  para_id,
                                  essay_para_id,
                                  para_type,
                                  para_stance,
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  essay_token_id,
                                  para_token_id
                                  )
                          )

                    essay_token_id += 1
                    para_token_id += 1
            else:
                para_type = "body"
                if should_use_tag:
                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t<para-body>"
                          .format(essay_id,
                                  para_id,
                                  essay_para_id,
                                  para_type,
                                  para_stance,
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  essay_token_id,
                                  para_token_id
                                  )
                          )

                    essay_token_id += 1
                    para_token_id += 1

            for ac in paragraph.ACs:
                if ac.tag == "O":
                    for token in ac.text.split():
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                              .format(essay_id,
                                      para_id,
                                      essay_para_id,
                                      para_type,
                                      para_stance,
                                      "-",
                                      "-",
                                      "-",
                                      "-",
                                      "-",
                                      essay_token_id,
                                      para_token_id,
                                      token
                                      )
                              )

                        essay_token_id += 1
                        para_token_id += 1
                else:
                    if ac.tag.split(":")[0] == "Premise":
                        ac_type = ac.tag.split(":")[0]
                        ac_parent = int(ac.tag.split(":")[1])
                        ac_relation_type = ac.tag.split(":")[2]
                    else:
                        ac_type = ac.tag
                        ac_parent = None
                        ac_relation_type = None

                    if should_use_tag:
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t<AC>"
                              .format(essay_id,
                                      para_id,
                                      essay_para_id,
                                      para_type,
                                      para_stance,
                                      essay_ac_id,
                                      para_ac_id,
                                      ac_type,
                                      ac_parent,
                                      ac_relation_type,
                                      essay_token_id,
                                      para_token_id
                                      )
                              )

                        essay_token_id += 1
                        para_token_id += 1

                    for token in ac.text.split():
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                              .format(essay_id,
                                      para_id,
                                      essay_para_id,
                                      para_type,
                                      para_stance,
                                      essay_ac_id,
                                      para_ac_id,
                                      ac_type,
                                      ac_parent,
                                      ac_relation_type,
                                      essay_token_id,
                                      para_token_id,
                                      token
                                      )
                              )

                        essay_token_id += 1
                        para_token_id += 1

                    if should_use_tag:
                        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t</AC>"
                              .format(essay_id,
                                      para_id,
                                      essay_para_id,
                                      para_type,
                                      para_stance,
                                      essay_ac_id,
                                      para_ac_id,
                                      ac_type,
                                      ac_parent,
                                      ac_relation_type,
                                      essay_token_id,
                                      para_token_id
                                      )
                              )

                        essay_token_id += 1
                        para_token_id += 1
                    essay_ac_id += 1
                    para_ac_id += 1

            if should_use_tag:
                if essay_para_id == 0:
                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t</prompt>"
                          .format(essay_id,
                                  para_id,
                                  essay_para_id,
                                  para_type,
                                  para_stance,
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  essay_token_id,
                                  para_token_id
                                  )
                          )
                elif essay_para_id == 1:
                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t</para-intro>"
                          .format(essay_id,
                                  para_id,
                                  essay_para_id,
                                  para_type,
                                  para_stance,
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  essay_token_id,
                                  para_token_id
                                  )
                          )
                elif essay_para_id == para_n - 1:
                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t</para-conclusion>"
                          .format(essay_id,
                                  para_id,
                                  essay_para_id,
                                  para_type,
                                  para_stance,
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  essay_token_id,
                                  para_token_id
                                  )
                          )
                else:
                    print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t</para-body>"
                          .format(essay_id,
                                  para_id,
                                  essay_para_id,
                                  para_type,
                                  para_stance,
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  "-",
                                  essay_token_id,
                                  para_token_id
                                  )
                          )

                essay_token_id += 1
                para_token_id += 1
            para_id += 1

        if should_use_tag:
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t</essay>"
                  .format(essay_id,
                          "-",
                          "-",
                          "-",
                          "-",
                          "-",
                          "-",
                          "-",
                          "-",
                          "-",
                          essay_token_id,
                          para_token_id
                          )
                  )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", action="store_true")

    args = parser.parse_args()
    main(args.tag)
