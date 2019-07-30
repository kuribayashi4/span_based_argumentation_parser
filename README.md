# An Empirical Study of Span Representation in Argumentation Structure Parsing
## Citation
```
@InProceedings{P19-1464,
  author = "Kuribayashi, Tatsuki
	and Ouchi, Hiroki
	and Inoue, Naoya
	and Reisert, Paul
    	and Miyoshi, Toshinori
    	and Suzuki, Jun
    	and Inui, Kentaro"
  title = 	"An Empirical Study of Span Representation in Argumentation Structure Parsing",
  booktitle = 	"Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
  year = 	"2019",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"4691-4698",
  location = 	"Florence, Italy",
  url = 	"https://www.aclweb.org/anthology/P19-1464"
}
```
- conference paper: https://www.aclweb.org/anthology/P19-1464

## Prerequirement
- python=3.6.5  
- cuda 9.0 (if the cuda version differs from your environment, change the cupy version in Pipfile)  
- pipenv  
- [GloVe(6B, 300d)](http://nlp.stanford.edu/data/glove.6B.zip)
  
## Installation
`pipenv shell`

## Preprocess
`bash preprocess.sh [DEVICE_ID]`
- Set DEVICE_ID if you use GPU for encoding the texts with ELMo. If not, set -1 to DEVICE_ID.

## Usage
If you run the training on CPU, unset the --device and -g options.  

### Experiments on the persuasive essay corpus
- ELMo, LSTM+dist  
`src/train.py --dataset PE --device -g [DEVICE_ID] --seed 39 --use-elmo 1 --elmo-path work/PE4ELMo.hdf5 --reps-type contextualized -ed 300 -hd 256 --optimizer Adam --lr 0.001 --ac-type-alpha 0.25 --link-type-alpha 0.25 --batchsize 16 --epoch 500 --dropout 0.5 --dropout-lstm 0.1 --lstm-ac --lstm-shell --lstm-ac-shell --lstm-type --elmo-layers avg -o [OUT_DIRECTORY]`  
  
- GloVe, LSTM+dist  
`src/train.py --dataset PE --device -g [DEVICE_ID] --seed 39 --glove-dir [GLOVE_PATH] --use-elmo 0 --reps-type contextualized  -ed 300 -hd 256 --optimizer Adam --lr 0.001 --ac-type-alpha 0.25 --link-type-alpha 0.25 --batchsize 16 --epoch 500 --dropout 0.5 --dropout-lstm 0.1 --lstm-ac --lstm-shell --lstm-ac-shell --lstm-type -o [OUT_DIRECTORY]`

### Training on the arg-microtext corpus (10 iterations of 5 fold-CV)
- ELMo, LSTM+dist  
`scripts/exp_MT.sh [DEVICE_ID] [OUT_DIRECTORY]`

### Check the performance of each model
`src/results.py -d [DIRECTORY_OF_OUTPUT_FILES]`

### Inference and analysis
`python src/infer_analysis.py -d [DIRECTORY_OF_OUTPUT_FILES] -g [DEVICE_ID] --device --data test`

## LICENSE
MIT License
