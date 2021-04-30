import torch
from torchtext.legacy import data, datasets
import nltk

nltk.download("words")


class DataLoad:
    """Download SST dataset and create dataloader using torchtext"""

    def __init__(self, sent_length=20, pretrained_vocab=None, batch_size=64):
        self.sent_length = sent_length
        WORDS = set(nltk.corpus.words.words())
        preprocess_pipeline = data.Pipeline(lambda x: x.lower()
                                            if x.lower() in WORDS else "<unk>")

        self.TEXT = data.Field(batch_first=True,
                               fix_length=self.sent_length,
                               preprocessing=preprocess_pipeline,
                               lower=True)
        # self.TEXT = data.Field(batch_first=True, fix_length=self.sent_length)
        self.LABEL = data.Field(sequential=False, dtype=torch.long)

        self.train, self.val, self.test = datasets.SST.splits(
            self.TEXT, self.LABEL, fine_grained=True, train_subtrees=False)

        # self.train = self._remove_non_english_words(self.train)
        # self.val = self._remove_non_english_words(self.val)
        # self.test = self._remove_non_english_words(self.test)

        # build vocab
        self.TEXT.build_vocab(self.train, vectors=pretrained_vocab)
        self.LABEL.build_vocab(self.train)

        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
            (self.train, self.val, self.test), batch_size=batch_size)

    def _remove_non_english_words(self, sentences):
        """Remove non English words and numeric"""
        WORDS = set(nltk.corpus.words.words())

        for sentence in sentences:
            old_sentence = vars(sentence)['text']
            new_sentence = [word for word in old_sentence if word in WORDS]
            vars(sentence)['text'] = new_sentence

        return sentences