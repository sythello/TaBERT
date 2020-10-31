#!/bin/bash

# CW_DATA_PATH=data/datasets/commoncrawl.sample.jsonl
# CW_DATA_PATH=/Users/mac/Desktop/syt/Deep-Learning/Dataset/TaBERT_datasets/CommonCrawl/raw/
CW_DATA_PATH=/vault/TaBERT_datasets/CommonCrawl/raw/

# OUTPUT_FILE=data/preprocessed_data/common_crawl.preprocessed.jsonl
# OUTPUT_FILE=/Users/mac/Desktop/syt/Deep-Learning/Dataset/TaBERT_datasets/CommonCrawl/common_crawl.preprocessed.jsonl
OUTPUT_FILE=/vault/TaBERT_datasets/CommonCrawl/common_crawl.preprocessed.jsonl

FILTER="0[0-9].tar.gz"
# FILTER="00.tar.gz"

python \
    -m preprocess.common_crawl \
    --worker_num 12 \
    --input_file ${CW_DATA_PATH} \
    --filter ${FILTER} \
    --output_file ${OUTPUT_FILE}
