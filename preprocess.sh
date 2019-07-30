cat data/acl2017-neural_end2end_am/data/conll/Essay_Level/train.dat data/acl2017-neural_end2end_am/data/conll/Essay_Level/test.dat data/acl2017-neural_end2end_am/data/conll/Essay_Level/dev.dat > data/acl2017-neural_end2end_am/data/conll/Essay_Level/all.dat
cat data/acl2017-neural_end2end_am/data/conll/Paragraph_Level/train.dat data/acl2017-neural_end2end_am/data/conll/Paragraph_Level/test.dat data/acl2017-neural_end2end_am/data/conll/Paragraph_Level/dev.dat > data/acl2017-neural_end2end_am/data/conll/Paragraph_Level/all.dat
echo 'Load persuasive essay corpus (PE)'
python src/preprocess/load_text_essays.py --tag > ./work/PE_data.tsv
echo 'Load arg-microtext corpus (MT)'
python src/preprocess/make_author_folds_argmicro.py > ./work/folds_author.json
python src/preprocess/load_text_argmicro.py --tag > ./work/MT_data.tsv

echo 'Preprocess for ELMo'
cat work/PE_data.tsv| python src/preprocess/make_data_for_elmo.py > work/PE4ELMo.tsv
cat work/MT_data.tsv| python src/preprocess/make_data_for_elmo.py > work/MT4ELMo.tsv

echo 'Encoding PEC with ELMo'
allennlp elmo work/PE4ELMo.tsv work/PE4ELMo.hdf5 --all --cuda-device $1
echo 'Encoding MTC with ELMo'
allennlp elmo work/MT4ELMo.tsv work/MT4ELMo.hdf5 --all --cuda-device $1

echo 'Make vocab file'
bash src/preprocess/make_vocab.sh work/PE4ELMo.tsv
bash src/preprocess/make_vocab.sh work/MT4ELMo.tsv

echo 'Finish!'
