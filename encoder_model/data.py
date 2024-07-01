" Data module for possible model train/test/eval dataset creator "

import logging
from typing import List, Dict, Any, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from encoder_model.model import LABELS

logger = logging.getLogger(__name__)


class ModelDataset(Dataset):
    """
    Dataset class for ModelDataset.

    Args:
        encodings (dict): A dictionary containing the input encodings (inpout_ids, attention_mask, token_type_ids).
        category_main (list): A list of main categories.
        intent (list): A list of sub categories.

    Returns:
        dict: A dictionary containing the input encodings, main category, and sub category for a specific index.

    """

    def __init__(self, encodings, category_main, intent):
        self.encodings = encodings
        self.category_main = category_main
        self.intent = intent

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        # item['category_main'] = self.category_main[idx].cpu()
        # item['intent'] = self.intent[idx].cpu()
        # item['labels'] = {"category_main": self.category_main[idx], "intent": self.intent[idx]}
        item["labels"] = torch.tensor([self.category_main[idx], self.intent[idx]])

        return item

    def __len__(self):
        return len(self.category_main)


class DataProcessor:
    "Processor for model train/test/eval dataset creator"

    def __init__(self, tokenizer: BertTokenizer, max_length: int = 64):

        self.tokenizer = tokenizer  # BertTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tokenize_text(self, texts: List[str]):
        """
        Tokenizes a list of texts using the tokenizer.

        Args:
            texts (List[str]): A list of texts to be tokenized.

        Returns:
            dict: A dictionary containing the tokenized texts, with the following keys:
                - 'input_ids': The tokenized input texts.
                - 'attention_mask': The attention mask indicating which tokens to attend to.
        """

        return self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            # add_special_tokens=True,
            return_tensors="pt",
        )

    def build_dataset(self, dataset: List[Dict[str, Union[str,Dict[Any,Any],List[str]]]], text_field: str):
        """
        Builds a dataset for training or evaluation.

        Args:
            dataset (List[Dict]): The input dataset containing data dictionaries. JSonl file. 
            text_field: (str): The input text key of dataset.

        Returns:
            ModelDataset: The built dataset object.

        """

        texts = []
        category_main_labels = []
        intent_labels = []

        for data in tqdm(dataset, desc="Processing Labels"):
            texts.append(data[text_field])
            # make labels a list
            category_main_labels.append(self.process_multiclass_label(label_key="category_main", label_value=data["labels"]["category_main"]))
            intent_labels.append(self.process_multiclass_label(label_key="intent", label_value=data["labels"]["intent"]))

        logger.info("Tokenizing Texts")
        tokenized_texts = self.tokenize_text(texts=texts)
        # need to unsqueeze the tensor to make it 2D
        # for key, value in tokenized_texts.items():
        #     # print(key, value)
        #     tokenized_texts[key] = torch.unsqueeze(value, 1)

        # Create dataset
        out_dataset = ModelDataset(tokenized_texts, category_main_labels, intent_labels)

        return out_dataset

    def process_multiclass_label(self, label_key: str, label_value):
        """
        Process a multiclass label by converting it to a class index.

        Args:
            label_key (str): The key of the label.
            label_value: The value of the label.

        Returns:
            torch.Tensor: The class index tensor.

        Note:
            The pytorch multiclass `torch.nn.CrossEntropyLoss` loss expects a class index as the target and not a one-hot encoded tensor.
        """

        value = LABELS[label_key].index(label_value)
        return torch.tensor(value)
