from torch.utils.data import Dataset
import datasets
from transformers import AutoTokenizer
import pickle
from matplotlib.pyplot import *

'''
Abstractive Summarization Dataset
dependencies: torch, transformers, datasets

params
 dataset types: cnndm, xsum, wiki
 min_length: int
 max_length: int (word count limit per text, automatic filtering based on this parameter if given)
 split: train, validation, test
 data_dir: path/to/dir/ (where the data directory located)
'''

# dataset = ASDataset(dataset_type="cnndm", split="train")


class ASDataset(Dataset):
    def __init__(
            self,
            dataset_type,
            min_length=30,
            max_length=1024,
            split="train",
            data_dir="",
            tokenizer=None):
        """ data format: text, summary """

        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.split = split

        self.reset_dataset()    # Loads the dataset
        self.max_length = max_length    # maximum token count
        self.min_length = min_length    # minimum word count

        if tokenizer is None:   # Sets the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "sshleifer/distilbart-xsum-1-1")
        else:
            self.tokenizer = tokenizer

        self.filter_by_token_count()
        # removes the instances with shorter texts than 30 words
        self.filter_by_min_word_count(self.min_length)

    # filters the dataset by valid data indices generated before
    def filter_by_token_count(self):
        valid_data_inds = pickle.load(
            open(
                self.data_dir +
                f"valid_data_inds/{self.dataset_type}_{self.split}_valid.p",
                "rb"))
        prev_len = self.dataset.num_rows
        self.dataset = self.dataset.filter(
            lambda example,
            idx: valid_data_inds[idx] == 1,
            with_indices=True)
        curr_len = self.dataset.num_rows
        print(
            "Dataset filtered based on max token length of 1024.\nThe dataset size decreased from {} to {}.".format(
                prev_len,
                curr_len))

    # creates valid data indices by the given max token count
    def extract_mask_info(self):
        self.valid_data_inds = []
        for text in self.dataset["text"]:
            self.valid_data_inds.append(1 - self.tokenizer(text,
                                                           truncation=True,
                                                           padding='max_length',
                                                           max_length=self.max_length,
                                                           return_tensors="pt")["attention_mask"][0][-1])

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_max_length(self):
        return self.max_length

    def filter_by_max_word_count(self, max_word):
        prev_len = self.dataset.num_rows
        self.dataset = self.dataset.filter(lambda example: len(
            example['text'].strip().split()) <= max_word, num_proc=4)
        curr_len = self.dataset.num_rows
        if prev_len == curr_len:
            print(
                "There is no texts longer than the threshold. Filtering based on max word count is done.")
        else:
            print(
                "Filtering based on max word count is done.\nThe dataset size decreased from {} to {}.".format(
                    prev_len,
                    curr_len))

    def filter_by_min_word_count(self, min_word):
        prev_len = self.dataset.num_rows
        self.dataset = self.dataset.filter(lambda example: len(
            example['text'].strip().split()) >= min_word, num_proc=4)
        curr_len = self.dataset.num_rows
        if prev_len == curr_len:
            print(
                "There is no texts shorter than the threshold. Filtering based on min word count is done.")
        else:
            print(
                "Filtering based on min word count is done.\nThe dataset size decreased from {} to {}.".format(
                    prev_len,
                    curr_len))

    # reloads the original dataset
    def reset_dataset(self):
        if self.dataset_type == "cnndm":
            self.dataset = datasets.load_dataset(
                "cnn_dailymail", "3.0.0", split=self.split)
            self.dataset = self.dataset.rename_column("article", "text")
            self.dataset = self.dataset.rename_column("highlights", "summary")
        elif self.dataset_type == "xsum":
            self.dataset = datasets.load_dataset("xsum", split=self.split)
            self.dataset = self.dataset.rename_column("document", "text")
        elif self.dataset_type == "wiki":
            self.dataset = datasets.load_from_disk(
                self.data_dir + f"data/wikipedia_{self.split}")
        self.dataset = self.dataset.filter(
            lambda example: len(
                example['summary']) <= len(
                example['text']),
            num_proc=4)

    # visualizes text-summary length comparison

    def visualize_wordcounts(self):
        def helper_func(example):
            example['text_wc'] = len(example['text'].strip().split())
            example['summ_wc'] = len(example['summary'].strip().split())
            example['ratio'] = example['summ_wc'] / example['text_wc']
            return example
        stats = self.dataset.map(
            helper_func, num_proc=4, remove_columns=[
                'text', 'summary'])
        stats = stats.sort('text_wc')
        moving_ave = np.zeros(stats.num_rows - 1)
        summaries = np.array(stats["summ_wc"])
        window = int(len(self) / 100)
        for i in range(window):
            moving_ave = moving_ave[1:] + summaries[:-(i + 2)]
        moving_ave = moving_ave / window
        plot(range(stats.num_rows)[(window + 1):],
             moving_ave, label='Summary Word Counts')
        title("Moving Average of Summary Size")
        ylabel('Word Count (averaged over window)')
        xlabel("Instances ordered by Text Size")
        legend()
        show()
        ave_ratio = sum(stats["ratio"]) / stats.num_rows
        ove_ratio = sum(stats['summ_wc']) / sum(stats['text_wc'])

        print(
            "\nThe average ratio between word counts of summary-text pairs is {}".format(ave_ratio))
        print("\nThe overall ratio between word counts of text and summaries is {}".format(
            ove_ratio))
