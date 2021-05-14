for file in ./data/sample/*
do
    python3 create_pretraining_data.py \
    --input_file=$file \
    --output_file=data/tfrecord/$(basename "${file%.*}").tfrecord \
    --vocab_file=./vocab.txt \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5 \
    --do_whole_word_mask=True \
    --do_lower_case=False
done
