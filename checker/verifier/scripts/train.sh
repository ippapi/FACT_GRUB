DATA_NAME=./checker/verifier/dummy
DATA_DIR=./$DATA_NAME

export PYTHONPATH=$(pwd)

python ./checker/verifier/main.py  \
    --data_name $DATA_NAME \
    --train_file $DATA_DIR/train.parquet \
    --dev_file $DATA_DIR/dev.parquet \
    --model_name "vinai/phobert-large" \
    --epochs 50 \
    --batch_size 32 \
    --lr 2e-05 \
    --logging_steps 25 \
    --save_steps 25 \
    --do_train 