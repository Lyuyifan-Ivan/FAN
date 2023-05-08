import random
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MMPNERProcessor(object):
    def __init__(self, data_path, bert_name) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    imgs.append(img_id)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

            comput_sent = sorted(raw_words, key=lambda x: len(x), reverse=True)
            print("The longest length of all data:", len(comput_sent[0]))

        assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}" \
            .format(len(raw_words), len(raw_targets), len(imgs))

        return {"words": raw_words, "targets": raw_targets, "imgs": imgs}

    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]",
                      "[SEP]"]
        label_mapping = {label: idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0
        return label_mapping


class MMPNERDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, max_seq=40, mode='train', ignore_idx=0) -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_from_file(mode)
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.mode = mode

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, label_list, img = self.data_dict['words'][idx], self.data_dict['targets'][idx], \
        self.data_dict['imgs'][idx]
        tokens, labels = [], []
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    labels.append(self.label_mapping["X"])
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[0:(self.max_seq - 2)]
            labels = labels[0:(self.max_seq - 2)]

        if self.img_path is not None:
            # image process
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                img_path = os.path.join(self.img_path, '17_06_4705.jpg')  # 'inf.png'
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)

        return word_list, image, labels

        # assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(labels)
        # return torch.tensor(input_ids), torch.tensor(token_type_ids), \
        #        torch.tensor(attention_mask), torch.tensor(labels), image


class Collate:
    def __init__(self, args, processor):
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        batch_text = [" ".join(b[0]) for b in batch_data]
        batch_img = torch.FloatTensor([np.array(b[1]) for b in batch_data])
        batch_label = [b[2] for b in batch_data]

        batch_text_encoder_dict = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_text,
                                                                   padding='longest',
                                                                   return_attention_mask=True,
                                                                   return_length=True,
                                                                   return_tensors='pt')

        self.batch_text_input_ids = batch_text_encoder_dict["input_ids"]
        self.batch_text_token_type_ids = batch_text_encoder_dict['token_type_ids']
        self.batch_text_attention_mask = batch_text_encoder_dict['attention_mask']
        self.batch_text_length = batch_text_encoder_dict['length']

        self.batch_label = [[self.label_mapping["[CLS]"]] + label + [self.label_mapping["[SEP]"]] +
                            [self.label_mapping["PAD"]] * (max(self.batch_text_length) - len(label) - 2)
                            for label in batch_label]

        assert len(self.batch_text_input_ids) == len(self.batch_text_attention_mask) \
               == len(self.batch_text_token_type_ids) == len(self.batch_label)

        return torch.tensor(self.batch_text_input_ids), torch.tensor(self.batch_text_token_type_ids), \
            torch.tensor(self.batch_text_attention_mask), torch.tensor(self.batch_label), batch_img
