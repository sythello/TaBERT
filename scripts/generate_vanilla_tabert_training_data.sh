#!/usr/bin/env bash
set +e

# output_dir=data/train_data/vanilla_tabert
output_dir=/Users/mac/Desktop/syt/Deep-Learning/Dataset/TaBERT_datasets/train_data/vanilla_tabert_sample_ac3
# output_dir=/vault/TaBERT_datasets/train_data/vanilla_tabert
mkdir -p ${output_dir}

train_corpus=/Users/mac/Desktop/syt/Deep-Learning/Dataset/TaBERT_datasets/tables_sample.jsonl
# train_corpus=/vault/TaBERT_datasets/tables.jsonl

word_confusion_path=data/word_confusions.pkl

python -m utils.generate_vanilla_tabert_training_data \
    --output_dir ${output_dir} \
    --train_corpus ${train_corpus} \
    --base_model_name bert-base-uncased \
    --do_lower_case \
    --epochs_to_generate 15 \
    --max_context_len 128 \
    --table_mask_strategy column \
    --context_sample_strategy concate_and_enumerate \
    --masked_column_prob 0.2 \
    --masked_context_prob 0.15 \
    --max_predictions_per_seq 200 \
    --cell_input_template 'column|type|value' \
    --column_delimiter "[SEP]" \
    --use_acoustic_confusion \
    --word_confusion_path ${word_confusion_path} \
    --include_ref_tokens \
    --add_fixing_in_mlm
    
    # --include_ref_tokens
    # --add_fixing_in_mlm