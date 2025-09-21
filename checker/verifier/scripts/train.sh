DATA_NAME=dummy
DATA_DIR=./$DATA_NAME

export PYTHONPATH=$(pwd)

python ./checker/verifier/main.py  \
    --data_name $DATA_NAME \
    --train_file $DATA_DIR/train.csv \
    --dev_file $DATA_DIR/dev.csv \
    --model_name "vinai/phobert-base" \
    --batch_size 32 \
    --lr 2e-05 \
    --logging_steps 200 \
    --save_steps 200 \
    --do_train 