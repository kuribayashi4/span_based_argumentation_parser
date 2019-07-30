import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", required=True)
parser.add_argument("-i", "--iterations", default=1000)
args = parser.parse_args()

files = glob.glob(args.directory + "result*")

dev_results = {}
dev_li = {}
dev_ltc = {}
dev_atc = {}
test_results = {}
test_li = {}
test_ltc = {}
test_atc = {}

for file in files:
    if "dev" in file:
        data_type = "dev"
    else:
        data_type = "test"

    dirname = file.split("/")[-2]
    with open(file) as f:
        for line in f:
            if line.strip():
                line = line.strip().split()
                if line[0] == "macro_f_link:":
                    macro_f_link = float(line[1])
                    f_link = float(line[3])
                    f_nolink = float(line[5])
                if line[0] == "macro_f_type:":
                    macro_f_type = float(line[1])
                    f_premise = float(line[3])
                    f_claim = float(line[5])
                    f_majorclaim = float(line[7])
                if line[0] == "f_macro_link_type:":
                    macro_f_link_type = float(line[1])
                    f_support = float(line[3])
                    f_attack = float(line[5])

        if data_type == "dev":
            dev_li = (macro_f_link, f_link, f_nolink)
            dev_atc = (macro_f_type, f_premise, f_claim, f_majorclaim)
            dev_ltc = (macro_f_link_type, f_support, f_attack)
            dev_overall = (macro_f_link + macro_f_type + macro_f_link_type)/3

            print()
            print("dev")
            print("Overall:\t{}".format(dev_overall))
            print("LINK IDENTIFICATION:\t{}".format("\t".join(str(score)
                                                              for score in dev_li)))
            print("LINK TYPE:\t{}".format(
                "\t".join(str(score) for score in dev_ltc)))
            print("AC TYPE :\t{}".format(
                "\t".join(str(score) for score in dev_atc)))
        else:
            test_li = (macro_f_link, f_link, f_nolink)
            test_atc = (macro_f_type, f_premise,
                                 f_claim, f_majorclaim)
            test_ltc = (macro_f_link_type, f_support, f_attack)
            test_overall = (macro_f_link + macro_f_type + macro_f_link_type)/3

            print()
            print("test")
            print("Overall:\t{}".format(test_overall))
            print("LINK IDENTIFICATION:\t{}".format("\t".join(str(score)
                                                              for score in test_li)))
            print("LINK TYPE:\t{}".format(
                "\t".join(str(score) for score in test_ltc)))
            print("AC TYPE :\t{}".format(
                "\t".join(str(score) for score in test_atc)))
