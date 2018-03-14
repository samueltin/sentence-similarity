from gensim.models import KeyedVectors
import os

ZH_WORD2VEC_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     '..', 'word_embedding', 'zh', 'fasttext', 'zh.vec')

EN_WORD2VEC_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     '..', 'word_embedding', 'en', 'word2vec', 'GoogleNews-vectors-negative300.bin')

# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)

model = KeyedVectors.load_word2vec_format(EN_WORD2VEC_PATH, binary=True, unicode_errors='ignore')

dog = model['dog']
print(dog.shape)
print(dog[:10])

# Some predefined functions that show content related information for given words
print(model.most_similar(positive=['woman', 'king'], negative=['man']))

print(model.doesnt_match("breakfast cereal dinner lunch".split()))
