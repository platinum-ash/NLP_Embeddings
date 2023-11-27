# -*- coding: utf-8 -*-


from os import path as os_path
from logging import getLogger
from typing import Dict, Callable
import torch
from torchtext.data import to_map_style_dataset
from torch.utils.data import DataLoader
from functools import partial

MIN_FREQUENCY_OF_WORD = 10
UNKNOWN_WORD_INDEX = 0
CBOW_N_WORDS = 2
SKIPGRAM_N_WORDS = 2
MAX_SEQUENCE_LENGTH = 100



class DataSet:
    """
    This class provides a simple API for loading the data set
    The preprocessing of the text will be handled in this class
    """

    def __init__(self, 
                 path_to_data_set: str = "datset.raw"):
        # Make sure that the provided file path is correct
        if not os_path.isfile(path_to_data_set):
            print(f"The provided file path to the data set is invalid : {path_to_data_set}")
            raise IOError("Unable to load data set file")

        self._data_set_path = path_to_data_set
        # Read the entire contents of the dataset to memory
        with open(self._data_set_path, mode="r") as open_dataset:
            self._loaded_data_set = open_dataset.read()

        self._data_set_lines = self._loaded_data_set.splitlines()

        # Now the entire data set is loaded into memory
        # Proceed to extract the vocabulary from the data set
        self._vocabulary = self.__extract_vocabulary()
        self._indexed_vocabulary = self.__get_indexed_vocabulary()

    def __extract_vocabulary(self) -> Dict[str, int]:
        """
        This method extracts all the unique words in the data set
        :return: The vocabulary as a dictionary with the frequency of words as values and words as keys
        """
        vocabulary = {}
        for line in self._loaded_data_set.splitlines():
            for word in line.split():
                if word not in vocabulary:
                    vocabulary[word] = 1
                else:
                    vocabulary[word] += 1

        print("Successfully built vocabulary from the data set")
        print(f"The total number of unique words found are : {len(vocabulary)}")

        return vocabulary

    def __get_indexed_vocabulary(self) -> Dict[str, int]:
        """
        This function maps each word to an index, while taking into account
        the minimum frequency that is needed for  a word to be included
        :return: The method returns a dictionary which maps each word to an index
        """
        indexed_vocabulary = {}
        index = 1
        for word in self._vocabulary.keys():
            if self._vocabulary[word] < MIN_FREQUENCY_OF_WORD:
                continue
            else:
                indexed_vocabulary[word] = index
                index += 1

        return indexed_vocabulary

    
    def __getitem__(self, index: int):
        return self._data_set_lines[index]

    def __len__(self):
        return len(self._data_set_lines)
    
    def get_vocabulary(self):
        return self._indexed_vocabulary

    
    def get_word_indices(self, sentence: str) -> list:
        """
        This method returns the indices of each word in the sentence
        An unknown word is given the default index 0
        :param sentence: The sentence from the data set as a string
        :return: The indices from the _indexed_vocabulary
        """
        indices = []
        for word in sentence.split():
            if word in self._indexed_vocabulary:
                indices.append(self._indexed_vocabulary[word])
            else:
                # The vocabulary does not contain the word
                indices.append(UNKNOWN_WORD_INDEX)

        return indices

    @staticmethod
    def collate_cbow(batch, get_word_indices: Callable):
        """
        Collate_fn for CBOW model to be used with Dataloader.
        `batch` is expected to be list of text paragraphs.

        Context is represented as N=CBOW_N_WORDS past words
        and N=CBOW_N_WORDS future words.

        Long paragraphs will be truncated to contain
        no more that MAX_SEQUENCE_LENGTH tokens.

        Each element in `batch_input` is N=CBOW_N_WORDS*2 context words.
        Each element in `batch_output` is a middle word.
        """
        batch_input, batch_output = [], []
        for text in batch:
            text_tokens_ids = get_word_indices(text)

            if len(text_tokens_ids) < CBOW_N_WORDS * 2 + 1:
                continue

            if MAX_SEQUENCE_LENGTH:
                text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

            for idx in range(len(text_tokens_ids) - CBOW_N_WORDS * 2):
                token_id_sequence = text_tokens_ids[idx: (idx + CBOW_N_WORDS * 2 + 1)]
                output = token_id_sequence.pop(CBOW_N_WORDS)
                input_ = token_id_sequence
                batch_input.append(input_)
                batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output


    @staticmethod
    def collate_skipgram(batch, get_word_indices: Callable):
        """
        Collate_fn for Skip-Gram model to be used with Dataloader.
        `batch` is going to be sentences from the data set

        Context is represented as N=SKIPGRAM_N_WORDS past words
        and N=SKIPGRAM_N_WORDS future words.

        Each element in `batch_input` is a middle word.
        Each element in `batch_output` is a context word.
        """
        batch_input, batch_output = [], []
        for text in batch:
            text_tokens_ids = get_word_indices(text)

            if len(text_tokens_ids) < SKIPGRAM_N_WORDS * 2 + 1:
                continue

            if MAX_SEQUENCE_LENGTH:
                text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]

            for idx in range(len(text_tokens_ids) - SKIPGRAM_N_WORDS * 2):
                token_id_sequence = text_tokens_ids[idx: (idx + SKIPGRAM_N_WORDS * 2 + 1)]
                input_ = token_id_sequence.pop(SKIPGRAM_N_WORDS)
                outputs = token_id_sequence

                for output in outputs:
                    batch_input.append(input_)
                    batch_output.append(output)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output

    @staticmethod
    def get_dataloader_and_vocab(
            model_name, batch_size, shuffle, vocab=None
    ):
        data_set = DataSet()
        if not vocab:
            vocab = data_set.get_vocabulary()

        text_pipeline = data_set.get_word_indices

        if model_name == "cbow":
            collate_fn = DataSet.collate_cbow
        elif model_name == "skipgram":
            collate_fn = DataSet.collate_skipgram
        else:
            raise ValueError("Choose model from: cbow, skipgram")

        dataloader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=partial(collate_fn, get_word_indices=text_pipeline),
        )
        return dataloader, vocab

import torch.nn as nn

EMBEDDING_VECTOR_DIM = 150
# Restrict the maximum value of the weights for a word to prevent them from becoming too large
EMBED_MAX_NORM = 1


class CbowModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_VECTOR_DIM,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBEDDING_VECTOR_DIM,
            out_features=vocab_size)

    def forward(self, input_features):
        x = self.embedding_layer(input_features)
        # For the CBOW approach we have to use the mean of the embedded context words
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class SkipGramModel(nn.Module):
    """
    Class to create a skip gram model
    """
    def __init__(self, vocab_size: int):
        super(SkipGramModel, self).__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_VECTOR_DIM,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBEDDING_VECTOR_DIM,
            out_features=vocab_size,
        )

    def forward(self, input_features):
        x = self.embedding_layer(input_features)
        # No need to take mean in case of a skip gram mode
        x = self.linear(x)
        return x

import numpy as np
import torch.optim
from tqdm import tqdm
# Create the train loop
def train_epoch(model, train_dataloader,optimizer, criterion, device = torch.device("cuda"), train_steps= 100):
        model.train()
        running_loss = []

        for i in tqdm(range(1000)):
          for i, batch_data in enumerate(train_dataloader, 1):
            inputs = batch_data[0].to(device)
            labels = batch_data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

            if i == train_steps:
                break

          epoch_loss = np.mean(running_loss)
          print(f"Epoch loss : {epoch_loss}")

# Create a CBOW model

data_set = DataSet()
vocab = data_set.get_vocabulary()

model = CbowModel(vocab_size=len(vocab))
model.to(torch.device("cuda"))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_criteria = torch.nn.CrossEntropyLoss()

train_epoch(model=model, 
            train_dataloader=DataSet.get_dataloader_and_vocab(model_name="cbow", batch_size=15, vocab=vocab, shuffle=False)[0],
            optimizer=optimizer, criterion=loss_criteria
            )

print(torch.cuda.is_available())

import numpy as np
import pandas as pd
import torch
import sys

from sklearn.manifold import TSNE
import plotly.graph_objects as go

# embedding from first model layer
embeddings = list(model.parameters())[0]
embeddings = embeddings.cpu().detach().numpy()

# normalization
norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
norms = np.reshape(norms, (len(norms), 1))
embeddings_norm = embeddings / norms
embeddings_norm.shape

# get embeddings
embeddings_df = pd.DataFrame(embeddings)

# t-SNE transform
tsne = TSNE(n_components=2)
embeddings_df_trans = tsne.fit_transform(embeddings_df)
embeddings_df_trans = pd.DataFrame(embeddings_df_trans)

# get token order
embeddings_df_trans.index = data_set._indexed_vocabulary.keys()

# if token is a number
is_numeric = embeddings_df_trans.index.str.isnumeric()

color = np.where(is_numeric, "green", "black")
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=embeddings_df_trans[0],
        y=embeddings_df_trans[1],
        mode="text",
        text=embeddings_df_trans.index,
        textposition="middle center",
        textfont=dict(color=color),
    )
)
fig.show()

