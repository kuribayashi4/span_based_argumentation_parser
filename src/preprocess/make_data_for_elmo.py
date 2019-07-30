import itertools
import sys
import tqdm

for para_id, lines in tqdm.tqdm(itertools.groupby(sys.stdin, key=lambda x:x.split("\t")[1])):
    lines = list(lines)
    if not (lines[0].split("\t")[0] == "Essay_id" or lines[0].split("\t")[1] == "-"):
        print(" ".join([line.split("\t")[-1].strip() for line in lines]))
