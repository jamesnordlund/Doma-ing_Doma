from datasets import Dataset, DatasetDict
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from random import shuffle, seed
import spacy
import string
import torch
from transformers import (AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, BertForMaskedLM,
                          DataCollatorForLanguageModeling,
                          RobertaTokenizer, RobertaForMaskedLM,
                          Trainer, TrainingArguments, set_seed)
from wordcloud import WordCloud


def split_df(df,
             test_size=.75):
    """Splits a dataset into train and test, with test_size proportion allocated to the latter"""
    rows = [i for i in range(df.shape[0])]
    shuffle(rows)
    return df.iloc[rows[0:int(test_size*df.shape[0])]], df.iloc[rows[int(test_size*df.shape[0]):]]


def refine_model(out_path):
    """Trains a language model with bert-base-uncased as the base and Doma publications as the corpus"""

    nlp = spacy.load('en_core_web_trf')

    df = {'file': [], 'sent_id': [], 'text': []}
    for f_name in os.listdir(out_path + '/corpus/'):
        with open(out_path + '/corpus/' + f_name, 'r', encoding='utf8') as f:
            doc = nlp(f.read())
        for sent in doc.sents:
            df['file'].append(f_name)
            df['sent_id'].append(len(df['sent_id']))
            df['text'].append(sent.text)
    df = pd.DataFrame(df)
    seed(0)
    df_train, df_test = split_df(df)

    model_ckpt = 'bert-base-uncased'
    model_roberta = 'roberta-base'

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=128, return_special_tokens_mask=True)

    for model in [model_roberta, model_ckpt]:
        tokenizer = AutoTokenizer.from_pretrained(model)

        ds = DatasetDict({
            'train': Dataset.from_pandas(df_train.reset_index(drop=True)),
            'test': Dataset.from_pandas(df_test.reset_index(drop=True)),
        })
        ds = ds.map(tokenize, batched=True)
        ds.set_format('torch')

        set_seed(0)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        data_collator.return_tensors = 'pt'

        training_args = TrainingArguments(
            output_dir=f'{model}-doma-128', per_device_train_batch_size=32,
            logging_strategy='epoch', evaluation_strategy='epoch', save_strategy='epoch',
            num_train_epochs=30, push_to_hub=False, log_level='error', report_to='none'
        )

        trainer = Trainer(
            model=AutoModelForMaskedLM.from_pretrained(model), tokenizer=tokenizer, args=training_args,
            data_collator=data_collator, train_dataset=ds['train'], eval_dataset=ds['test']
        )

        trainer.train()

        trainer.save_model(f'{model}-doma-128/over_fit')

        log = pd.DataFrame(trainer.state.log_history)
        log.dropna(subset=["eval_loss"]).reset_index()["eval_loss"].plot(label="Validation")
        log.dropna(subset=["loss"]).reset_index()["loss"].plot(label="Train")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")

        plt.savefig(f'{model}-doma-128/training_curve.png')
        plt.show()
        plt.close()


def make_word_cloud(out_path):
    """Creates a word cloud based on the corpus.  The corpus dir is not published, so save to model dir."""

    text = []
    for f_name in os.listdir(out_path + '/corpus/'):
        with open(out_path + '/corpus/' + f_name, 'r', encoding='utf8') as f:
            text.append(f.read())
    wordcloud = WordCloud(max_font_size=50, max_words=100).generate(' '.join(text))
    #plt.figure(figsize=(15, 15))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # never hurts to do a word cloud
    plt.savefig('corpus_word_cloud.png')
    plt.show()
    plt.close()


# function from: https://github.com/renatoviolin/next_word_prediction/blob/master/main.py
def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


# function from: https://github.com/renatoviolin/next_word_prediction/blob/master/main.py
def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


# modified from: https://github.com/renatoviolin/next_word_prediction/blob/master/main.py
def get_all_predictions(text_sentence,
                        top_clean=5):
    """Predict masked token in text_sentence, keeping top_clean best predictions."""

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased',
                                                 output_hidden_states=True, output_attentions=True)

    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=False)
    roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base',
                                                       output_hidden_states=True, output_attentions=True)

    doma_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased-doma-128/over_fit', do_lower_case=False)
    doma_model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased-doma-128/over_fit',
                                                      output_hidden_states=True, output_attentions=True)

    rodoma_tokenizer = AutoTokenizer.from_pretrained('roberta-base-doma-128/over_fit', do_lower_case=False)
    rodoma_model = AutoModelForMaskedLM.from_pretrained('roberta-base-doma-128/over_fit',
                                                      output_hidden_states=True, output_attentions=True)

    lms = {'bert': [bert_tokenizer, bert_model],
           'roberta': [roberta_tokenizer, roberta_model],
           'doma': [doma_tokenizer, doma_model],
           'rodoma': [rodoma_tokenizer, rodoma_model]}
    output = {}

    for mod, lm in lms.items():
        input_ids, mask_idx = encode(lm[0], text_sentence)
        with torch.no_grad():
            predict = lm[1](input_ids)[0]
        output.update({mod: decode(lm[0], predict[0, mask_idx, :].topk(top_clean).indices.tolist(), top_clean).split()})

    return output


def word_prediction():
    sentences = [
        'Doma wants the closing process to be <mask>.',
        'The traditional mortgage process is too <mask>.',
        'Our approach to titling is <mask> for home buyers.',
        'With the aid of <mask>, we make closing faster.'
    ]
    results = {}
    for sent in sentences:
        results.update({sent: get_all_predictions(text_sentence=sent, top_clean=2)})
    with open('word_predictions.json', 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)