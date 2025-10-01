FILE_NAME=$1
MODEL_PATH=$2

export PYTHONPATH=$(pwd)

python ./checker/filter/main.py  \
    --use_leven --leven_threshold 0.3 \
    --use_cls --cls_threshold 0.2 --min_prob 0.8 \
    --model_name $MODEL_PATH\
    --input_file $FILE_NAME