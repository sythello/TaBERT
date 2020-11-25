import sys
import os
import re
import random
import ujson
import numpy as np
import pickle

from typing import List, Dict

from collections import defaultdict, Counter

from table_bert.utils import BertTokenizer

from transformers import GPT2Tokenizer, GPT2LMHeadModel


def detokenize_BertTokenizer(sentence: List[str]) -> str:
    return (
        " ".join(sentence)
        .replace(" ##", "")
        # .replace(" .", ".")
        # .replace(" ?", "?")
        # .replace(" !", "!")
        # .replace(" ,", ",")
        # .replace(" ' ", "'")
        # .replace(" n't", "n't")
        # .replace(" 'm", "'m")
        # .replace(" 's", "'s")
        # .replace(" 've", "'ve")
        # .replace(" 're", "'re")
    )
    
    ## These modifications can cause subword length mismatch


class SentenceAcousticConfuser(object):
    def __init__(self,
                 word_confusions_path: str,
                 default_p: float):
                 # fix_subword_lengths: bool = True,  # confusion word have same number of subwords as original word 
                 # bert_tokenizer_name: str = 'bert-base-uncased'

        assert 0 <= default_p <= 1
        self.word_confusions_path = word_confusions_path
        self.default_p = default_p
        
        with open(self.word_confusions_path, 'rb') as f:
            self.word_confusions = pickle.load(f)
    
    def sentence_confuse(self, sentence: List[str], p: float = None) -> List[str]:
        raise NotImplementedError


class SentenceAcousticConfuser_RandomReplace(SentenceAcousticConfuser):
    def sentence_confuse(self, sentence: List[str], p: float = None) -> List[str]:
        if p is None:
            p = self.default_p
        assert 0 <= p <= 1
    
        sen_len = len(sentence)
        confs_cnt = int(p * sen_len)

        confusable_positions = []
        for pos in range(sen_len):
            word = sentence[pos].upper()
            if len(self.word_confusions[word]) > 0:
                confusable_positions.append(pos)

        if len(confusable_positions) <= confs_cnt:
            # not enough positions for confusion
            confs_positions = confusable_positions
        else:
            confs_positions = random.sample(confusable_positions, k=confs_cnt)

        confs_sentence = list(sentence)
        for pos in confs_positions:
            word = sentence[pos].upper()
            confs_word = random.choice(list(self.word_confusions[word])).lower()
            if pos == 0:
                confs_word = confs_word.capitalize()
            confs_sentence[pos] = confs_word

        return confs_sentence


class SentenceAcousticConfuser_GPT2LossReplace(SentenceAcousticConfuser):
    def __init__(self,
                 word_confusions_path: str,
                 default_p: float):
        
        super().__init__(word_confusions_path, default_p)

        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model_lm = GPT2LMHeadModel.from_pretrained('gpt2')
    
    def _gpt2_lm_loss(self, sentence: List[str]) -> float:
        inputs = self.gpt2_tokenizer(' '.join(sentence), return_tensors="pt")
        outputs = self.gpt2_model_lm(**inputs, labels=inputs["input_ids"])
        loss = outputs[0].item()
        return loss
    
    def sentence_confuse(self, sentence: List[str], p: float = None) -> List[str]:
        if p is None:
            p = self.default_p
        assert 0 <= p <= 1
    
        sen_len = len(sentence)
        confs_cnt = int(p * sen_len)

        confusable_positions = []
        for pos in range(sen_len):
            word = sentence[pos].upper()
            if len(self.word_confusions[word]) > 0:
                confusable_positions.append(pos)

        if len(confusable_positions) <= confs_cnt:
            # not enough positions for confusion
            confs_positions = confusable_positions
        else:
            confs_positions = sorted(random.sample(confusable_positions, k=confs_cnt))

        confs_sentence = list(sentence)
        for pos in confs_positions:
            word = sentence[pos].upper()
            assert len(self.word_confusions[word]) > 0

            if len(self.word_confusions[word]) == 1:
                # No need for LM loss 
                _cw = next(iter(self.word_confusions[word])).lower()
                if pos == 0:
                    _cw = _cw.capitalize()
                confs_sentence[pos] = _cw
                continue

            # Compare different confusion words by their LM losses 
            best_lm_loss = np.inf
            best_confs_word = None
            for _confs_word in self.word_confusions[word]:
                _cw = _confs_word.lower()
                if pos == 0:
                    _cw = _cw.capitalize()

                _confs_sen = list(confs_sentence)
                _confs_sen[pos] = _cw

                _loss = self._gpt2_lm_loss(_confs_sen)
                # print(_confs_sen, _loss)

                if _loss < best_lm_loss:
                    best_lm_loss = _loss
                    best_confs_word = _cw

            assert best_confs_word is not None
            confs_sentence[pos] = best_confs_word

        return confs_sentence
    
if __name__ == '__main__':
    word_confusions_path = '/Users/mac/Desktop/syt/Deep-Learning/Repos/TaBERT/data/word_confusions.pkl'

    confuser_random = SentenceAcousticConfuser_RandomReplace(word_confusions_path, default_p=0.15)
    confuser_gpt2 = SentenceAcousticConfuser_GPT2LossReplace(word_confusions_path, default_p=0.15)

    sentence = ["What", "are", "the", "ids", "of", "the", "TV", "channels", "that", "do", "not", \
        "have", "any", "cartoons", "directed", "by", "Ben", "Jones", "?"]

    print(confuser_random.sentence_confuse(sentence))
    print(confuser_gpt2.sentence_confuse(sentence))

    print(confuser_random.sentence_confuse(sentence, p=0.5))
    print(confuser_gpt2.sentence_confuse(sentence, p=0.5))











