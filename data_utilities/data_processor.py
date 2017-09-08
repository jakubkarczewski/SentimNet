import numpy as np
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import string

# All chars we will work with
ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\\|_@#$%^&*~`+' \
           '-=<>()[]{} '
ALPHABET_LEN = len(ALPHABET)
# All chars => number mappings
DICT = {char: number for number, char in enumerate(ALPHABET)}

TEST_SET = 'data/test_set.csv'
VALID_SET = 'data/valid_set.csv'
TRAIN_SET = 'data/train_set.csv'
PROJECT_PATH = '/home/kuba/Development/SentNet/'
VALIDATION_PERCENTAGE = 0.03


class CorpusProcessor(object):

    def __init__(self, file, max_word_len):
        # Here we pass .csv file with data
        self.file = file
        # Max word length necessary to have a valid tensor later on
        self.max_word_len = max_word_len

    def encode_sentence(self, sentence):
        # Convert sentences to tensors of shape:
        # [ALPHABET_LEN x word length x sentence length

        sent = []
        # Need to remember the max value for sentence len in order
        # to later provide with valid zero-padding
        SENT_LEN = 0

        # Remove non ASCI chars
        filtered_sentence = filter(lambda x: x in string.printable, sentence)

        # word_tokenize() is a NLTK method that slits sentences into
        # valid tokens ie. words. (unidecode converts chars to utf-8)

        # This loop iterates through words
        for word in word_tokenize(unidecode(filtered_sentence)):

            # Encode word as binary vector
            word_encoded = np.zeros(shape=(self.max_word_len, ALPHABET_LEN))

            # This loop iterates through chars
            for i, char in enumerate(word):

                try:
                    char_code = DICT[char]
                    char_vec = np.zeros(ALPHABET_LEN)
                    # Fill the proper place in vectorized char with 1
                    char_vec[char_code] = 1
                    word_encoded[i] = char_vec

                except Exception as e:
                    print "Char is not in alphabet."
            # Append vectorized word
            sent.append(np.array(word_encoded))
            # Increment word number in a sentence
            SENT_LEN += 1

# TODO: INVESTIGATE HARD - what is yield, next, how the hell it loads to RAM

    def build_minibatch(self, sentences):
        # Stack a few sentences together and vectorize the sentiment score
        # 0: negative , 1: positive

        # For encoded text corpus
        minibatch_i = []
        # For sentiment vec
        minibatch_j = []
        max_len = 0

        for sentence in sentences:
            minibatch_j.append(np.array([0, 1]) if sentence[:1] == '0'
                               else np.array([1,0]))

            # Encode a particular sentence and get it's length
            sentence_encoded, sentence_len = self.encode_sentence(sentence)

            # Update the max sentence length
            if sentence_len > max_len:
                max_len = sentence_len

            # Append each sentence to our minibatch
            minibatch_i.append(sentence_encoded)


        # TODO: Understand the numpy magic BETTER

        # Now we need to fill 'holes' in our tensor which appeared because of
        # different sentence lengths
        lengths = np.array([len(i) for i in minibatch_i])

        # Create mask of valid spaces in our tensor
        mask = np.arrange(lengths.max()) < lengths[:, None]

        # Setup output array and insert vaild data
        output = np.zeros(shape=(mask.shape + (self.max_word_len,
                                               ALPHABET_LEN)), dtype='float32')
        output[mask] = np.concatenate(minibatch_i)

        # End of TODO scope ----------------------------------------------------

        return minibatch_i, np.array(minibatch_j)

    def load_to_ram(self, batch_size):
        # Loads n rows from file f to RAM
        self.data = []
        n_rows = batch_size
        while n_rows > 0:
            self.data.append(next(self.file))
            n_rows=-1
        if n_rows == 0:
            return True
        else:
            return False

    def iterate(self, batch_size, dataset=TRAIN_SET):

        # Just an estimate
        if dataset == TRAIN_SET:
            n_samples = 1600000 * (1-VALIDATION_PERCENTAGE)
        elif dataset == VALID_SET:
            n_samples = 1600000 * VALIDATION_PERCENTAGE
        elif dataset == TEST_SET:
            n_samples = 498

        # TODO: maybe we need double slash idk
        n_batch = int(n_samples / batch_size)

        # Create a minibatch, load to RAM and feed it until buffer empties
        for i in range(n_batch):
            if self.load_to_ram(batch_size)
                inputs, targets = self.build_minibatch(self.data)
                yield inputs, targets
        # end of TODO SCOPE -----------------------------------------------------------








