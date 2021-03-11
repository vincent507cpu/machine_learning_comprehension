import os
import tqdm
import pickle
import numpy as np
import json

import config
from utils import tokenzier, clean_text, word_tokenizer, buid_vocab, build_embedding, convert_idx

class Preprocessor:
    def __init__(self, data_dir, train_filename, dev_filenamre, tokenizer):
        self.data_dir = data_dir
        self.train_filename = train_filename
        self.dev_filenamre = dev_filenamre
        self.tokenizer = tokenizer

    def load_data(self, filename='train-2.0.json'):
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath) as f:
            self.data = json.load(f)

    def split_data(self, filename):
        self.load_data(filename)
        sub_dir = filename.split('-')[0]

        # create a subdirectory for Train and Dev data
        if not os.path.exists(os.path.join(self.data_dir, sub_dir)):
            os.makedirs(os.path.join(self.data_dir, sub_dir))

        with open(os.path.join(self.data_dir, sub_dir, sub_dir + '_context.txt'), 'w', encoding="utf-8") as context_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '_question.txt'), 'w', encoding="utf-8") as question_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '_answer.txt'), 'w', encoding="utf-8") as answer_file,\
             open(os.path.join(self.data_dir, sub_dir, sub_dir + '_labels.txt'), 'w', encoding="utf-8") as labels_file:

            # loop over the data
            for article_id in tqdm.tqdm(range(len(self.data['data']))):
                list_paragraphs = self.data['data'][article_id]['paragraphs']

                # loop over the paragraphs
                for paragraph in list_paragraphs:
                    context = paragraph['context']
                    context = clean_text(context)
                    context_tokens = [w for w in word_tokenize(context) if w]
                    spans = convert_idx(context, context_tokens)
                    qas = paragraph['qas']

                    # loop over Q/A
                    for qa in qas:
                        question = qa['question']
                        question = clean_text(question)
                        question_tokens = [w for w in word_tokenizer(question) if w]
                        if sub_dir == 'train':
                            answer_ids = 1 if qa['answers'] else 0
                        else:
                            answer_ids = len(qa['answers'])
                        labels = []

                        if answer_ids:
                            for answer_id in range(answer_ids):
                                answer = qa['answer'][answer_id]['text']
                                answer = clean_text(answer)
                                answer_tokens = [w for w in word_tokenizer(answer) if w]
                                answer_start = qa['answers'][answer_id]['answer_start']
                                answer_stop = answer_start + len(answer)
                                answer_span = []

                                for idx, span in enumerate(spans):
                                    if not (answer_stop <= span[0] or answer_start >= span[1]):
                                        answer_span.append(idx)
                                if not answer_span:
                                    continue
                                
                                labels.append(str(answer_span[0]) + ' ' + str(answer_span[-1]))
                            
                            # write to file
                            context_file.write(' '.join([token for token in context_tokens]) + '\n')
                            question_file.write(' '.join([token for token in question_tokens]) + '\n')
                            answer_file.write(' '.join([token for token in answer_tokens]) + '\n')
                            labels_file.write("|".join(labels) + "\n")

    def preprocess(self):
        self.split_data(train_filename)
        self.split_data(dev_filename)

    def extract_features(self, max_len_context=config.max_len_context, max_len_question=config.max_len_question,
                         max_len_word=config.max_len_word, is_train=True):
        # choose the right directory
        directory = 'train' if is_train else 'dev'

        # load context
        with open(os.path.join(self.data_dir, sub_dir, sub_dir + '_context.txt'), 'r', encoding="utf-8") as c:
            context = c.readlines()

        # load questions
        with open(os.path.join(self.data_dir, sub_dir, sub_dir + '_question.txt'), 'r', encoding="utf-8") as q:
            question = q.readlines()

        # load answer
        with open(os.path.join(self.data_dir, sub_dir, sub_dir + '_labels.txt'), 'r', encoding='utf-8') as l:
            lanels = l.readlines()

        