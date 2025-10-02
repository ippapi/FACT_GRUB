DATA_NAME=./checker/verifier/dummy
DATA_DIR=./$DATA_NAME

export PYTHONPATH=$(pwd)

python ./checker/verifier/main.py  \
    --data_name $DATA_NAME \
    --train_file $DATA_DIR/train.parquet \
    --dev_file $DATA_DIR/dev.parquet \
    --model_name "VietAI/vit5-base" \
    --epochs 3 \
    --batch_size 32 \
    --lr 2e-05 \
    --logging_steps 1 \
    --save_steps 1 \
    --do_train 