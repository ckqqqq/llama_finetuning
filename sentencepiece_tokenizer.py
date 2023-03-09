# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 12:45


from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import os


logger = getLogger()


class SentencePieceTokenizer:
    def __init__(self, tokenizer_name: str,**kwargs):
        model_path = tokenizer_name
        self.tokenizer_kwargs = kwargs
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_token_id: int = self.sp_model.bos_id()
        self.eos_token_id: int = self.sp_model.eos_id()
        self.pad_token_id: int = self.sp_model.pad_id()
        self.unk_token_id: int = self.sp_model.unk_id()
        self.sep_token_id : int = self.sp_model.eos_id()

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_token_id} - EOS ID: {self.eos_token_id}"
        )
        logger.info(
            f"PAD ID: {self.pad_token_id} - UNK ID: {self.unk_token_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

        # if self.pad_token_id < 0:
        #     self.pad_token_id = 0


    @classmethod
    def from_pretrained(cls,tokenizer_name,**kwargs):
        return cls(tokenizer_name,**kwargs)

    def encode(self, text: str, bos: bool=True, eos: bool=True) -> List[int]:
        assert type(text) is str
        t = self.sp_model.encode(text)
        if bos:
            t = [self.bos_token_id] + t
        if eos:
            t = t + [self.eos_token_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
