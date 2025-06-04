import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, device='mps' if torch.backends.mps.is_available() else 'cpu'):
        self.data = torch.tensor(data, dtype=torch.long, device=device)
        self.labels = torch.tensor(labels, dtype=torch.long, device=device)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        mask = (x != 0)
        mask = mask.to(dtype=torch.bool)
        mask = mask.to(device=self.data.device)
        return x, y, mask

class ImdbDatabase:
    def __init__(self, num_words: int=5000, max_len: int=100, batch_size: int = 32, data_file_name: str="/data/imdbdatabase.csv", device = 'mps' if torch.backends.mps.is_available() else 'cpu'):
        self.num_words = num_words
        self.max_len = max_len
        self.batch_size = batch_size
        self.data_file_name = data_file_name
        self.device = device
        # Load and initialize words dictionary
        self.word_index = tf.keras.datasets.imdb.get_word_index()
        self.word_index = {k: (v + 3) for k, v in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2
        self.word_index["<UNUSED>"] = 3
        self.word_index = {word: index for word, index in self.word_index.items() if index < 5000}

    def load_data(self, r_val: float=0.1, r_test: float=0.1):
        df = pd.read_csv(self.data_file_name)
        seq_name = ['s_' + str(i) for i in range(0, self.max_len)]
        x = df[seq_name].values.astype(float)
        y = df['target'].values.astype(int)
        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=r_test, random_state=42, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=r_val/(1-r_test), random_state=42, stratify=y_train_val)
        train_dataset = IMDBDataset(x_train, y_train, device=self.device)
        val_dataset = IMDBDataset(x_val, y_val, device=self.device)
        test_dataset = IMDBDataset(x_test, y_test, device=self.device)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def create_database(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=self.num_words, maxlen=self.max_len)
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=self.max_len, truncating='post')
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=self.max_len, truncating='post')
        x_data = np.concatenate((x_train, x_test), axis=0)
        y_data = np.concatenate((y_train, y_test), axis=0)
        data = {}
        for i in range(x_data.shape[1]):
            title = 's_' + str(i)
            data[title] = x_data[:, i]
        data['target'] = y_data
        data = pd.DataFrame(data)
        data.to_csv(self.data_file_name, index=False)

    def text_to_sequence(self, text):
        words = text.lower().split()
        sequence = [self.word_index.get(word, 2) for word in words]
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=self.max_len, truncating='post')
        return padded_sequence

    def preprocessing_text(self, text):
        padded_sequence = self.text_to_sequence(text)
        inputs_ids = torch.tensor(padded_sequence, dtype=torch.long)
        mask = (inputs_ids != 0)
        mask = mask.to(dtype=torch.bool)
        return inputs_ids, mask

    def batch_preprocessing_text(self, batch):
        inputs_ids, masks = [], []
        for text in batch:
            model_inputs = self.preprocessing_text(text)
            inputs_ids.append(model_inputs[0])
            masks.append(model_inputs[1])
        inputs_ids, masks = np.array(inputs_ids), np.array(masks)
        return torch.tensor(inputs_ids, dtype=torch.long).squeeze(1), torch.tensor(masks, dtype=torch.bool).squeeze(1)