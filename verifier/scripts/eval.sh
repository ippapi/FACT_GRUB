DATA_NAME=dummy
DATA_DIR=./$DATA_NAME
MODEL_PATH = $1

!python  ../main.py \
    --dev_file $DATA_DIR/dev.csv \
    --model_name MODEL_PATH \
    --resume \
    --do_eval