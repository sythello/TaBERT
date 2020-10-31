#!/bin/bash

# path to the jdk
# JAVA_PATH=/System/Library/Frameworks/JavaVM.framework/Versions/Current/Commands/
JAVA_PATH=/usr/bin/java/
CLASSPATH=contrib/wiki_extractor/tableBERT-1.0-SNAPSHOT-jar-with-dependencies.jar

# The following dump is a sample downloaded from
# https://dumps.wikimedia.org/enwiki/20200901/enwiki-20200901-pages-articles-multistream1.xml-p1p30303.bz2
# You may need to use the full Wikipedia dump for data extraction, for example,
# https://dumps.wikimedia.org/enwiki/20201001/enwiki-20201001-pages-articles-multistream.xml.bz2 .
# In our paper we used the dump `enwiki-20190520-pages-articles-multistream.xml.bz2`.

# wget -nc https://dumps.wikimedia.org/enwiki/20200901/enwiki-20200901-pages-articles-multistream1.xml-p1p30303.bz2 -P data/datasets/
# wget -nc https://dumps.wikimedia.org/enwiki/20201001/enwiki-20201001-pages-articles-multistream.xml.bz2 \
# -P /vault/TaBERT_datasets/WikiTables/raw/

# WIKI_DUMP=/Users/mac/Desktop/syt/Deep-Learning/Dataset/TaBERT_datasets/WikiTables/raw/enwiki-20201001-pages-articles-multistream.xml.bz2
WIKI_DUMP=/vault/TaBERT_datasets/WikiTables/raw/enwiki-20201001-pages-articles-multistream.xml.bz2

# OUTPUT_FILE=data/preprocessed_data/wiki_tables.jsonl
# OUTPUT_FILE=/Users/mac/Desktop/syt/Deep-Learning/Dataset/TaBERT_datasets/WikiTables/wiki_tables.jsonl
OUTPUT_FILE=/vault/TaBERT_datasets/WikiTables/wiki_tables.jsonl

CLASSPATH=${CLASSPATH} JAVA_PATH=${JAVA_PATH} python \
    -m preprocess.extract_wiki_data \
    --wiki_dump ${WIKI_DUMP} \
    --output_file ${OUTPUT_FILE}
