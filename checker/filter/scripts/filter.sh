for mode in train dev
do  
    input_file= ...
    python ../models/filter_synthetic_data.py \
            --use_leven --leven_threshold 0.3 \
            --use_cls --cls_threshold 0.2 --min_prob 0.8 \
            --model_name_or_path ...\
            --input_file $input_file
done