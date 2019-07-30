GPU=$1
OUT=$2

for i in 0 1 2 3 4 5 6 7 8 9
do
    for j in 0 1 2 3 4
    do
        python src/train.py \
                --dataset MT \
                --iteration $i \
                --fold $j \
                --device \
                -g $GPU \
                --seed 39 \
                --ac-type-alpha 0.25 \
                --link-type-alpha 0.25 \
                --decoder proposed \
                --use-elmo 1 \
                --elmo-path work/MT4ELMo.hdf5 \
                --elmo-layers avg \
                --dropout 0.9 \
                --dropout-lstm 0.9 \
                --lstm-ac \
                --lstm-shell \
                --lstm-ac-shell \
                --lstm-type \
                -ed 300 \
                -hd 256 \
                --epoch 4000 \
                --batchsize 16 \
                --optimizer Adam \
                --lr 0.001 \
                -o $OUT
    done
done

