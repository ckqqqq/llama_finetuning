# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os

import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.LLaMA_parallel import TransformerLLaMALMHeadModel, setup_model_parallel, LLaMAConfig
from transformers import HfArgumentParser

import generator
from data_utils import train_info_args, NN_DataHelper
from sentencepiece_tokenizer import SentencePieceTokenizer


class MyTransformer(TransformerLLaMALMHeadModel, with_pl=True):
    def __init__(self, *args, **kwargs):
        super(MyTransformer, self).__init__(*args, **kwargs)




if __name__ == '__main__':
    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>""",
    ]

    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_dict(train_info_args)



    # 并行
    setup_model_parallel()
    torch.manual_seed(1)


    dataHelper = NN_DataHelper(model_args, training_args, data_args)

    tokenizer, config, label2id, id2label = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=SentencePieceTokenizer, config_class_name=LLaMAConfig)


    #修改配置
    def modify_config(config):
        # 小参数
        config.inference = True
        config.max_seq_len = 512


    #加载新训练权重
    train_weight = './best_ckpt/best.pt'
    if os.path.exists(train_weight):
        config = LLaMAConfig.from_pretrained('./best_ckpt')
        modify_config(config)
        model = MyTransformer.load_from_checkpoint(train_weight, config=config,
                                                   model_args=model_args,
                                                   training_args=training_args)
    else:
        modify_config(config)
        config.n_layer = 16 # 官方默认32层 ， 使用小层推理
        model = MyTransformer(config=config, model_args=model_args, training_args=training_args)

    model.eval()
    model.half()
    model.to(torch.device('cuda:0'))



    # 预测
    with torch.inference_mode():
        results = generator.Generate(model=model.backbone, tokenizer=tokenizer,
                                     device=torch.device('cuda:0')).generate(
            prompts, max_gen_len=256, temperature=0.8, top_p=0.95,
        )

        for result in results:
            print(result)
            print("\n==================================\n")

