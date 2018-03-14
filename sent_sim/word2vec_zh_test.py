from gensim.models import KeyedVectors
import os
import numpy as np
from scipy import spatial



ZH_WORD2VEC_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     '..', 'word_embedding', 'zh', 'fasttext', 'zh.vec')

PRODUCT_MAP = {
    'iphone_8_plus_case': [
        ['蘋果', '8', 'plus', '手機', '殼'],
        ['iphone', '8', 'plus', '手機', '保護', '殼'],
        # ['iphone', '8', 'plus', 'case']
    ],
    'wireless_bluetooth_earphone': [
        ['藍', '牙', '無線', '耳機'],
        ['專業', '藍', '牙', '無線', '耳機'],
        # ['Wireless', 'Bluetooth', 'earphone']
    ],
    'apple_wireless_charger': [
        ['iphone', '8', '無線', '充電'],
        # ['Apple', 'wireless', 'charger', 'for', 'iphone', 'X'],
        ['蘋果', '無線', '充電', '器'],
        ['無線', '充電', '器', '蘋果']
    ]
}

TEST = ['蘋果', '充電', '機', '無線']


# Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
word_embedding = KeyedVectors.load_word2vec_format(ZH_WORD2VEC_PATH, binary=False, unicode_errors='ignore')

def _words2embedding(words):
    embeddings = []
    for word in words:
        try:
            embedding = word_embedding[word]
            embeddings.append(embedding)
        except KeyError as e:
            print("{} is missing in word embedding".format(word))
    return np.array(embeddings, dtype=np.float32)

def _check_product_map(product_map):
    for _, items in product_map.items():
        for item in items:
            embeddings = _words2embedding(item)
            mean_embed = _mean_embedding(embeddings)
            print ('test')

def _find_max_sim(target, product_map):
    max_sim = -999.99
    max_type = None
    max_item = None
    target_embed = _words2embedding(target)
    target_mean = _mean_embedding(target_embed)
    for type, items in product_map.items():
        for item in items:
            embeddings = _words2embedding(item)
            mean_embed = _mean_embedding(embeddings)
            curr_sim = _cal_cosine_sim(target_mean, mean_embed)
            if curr_sim > max_sim:
                max_sim = curr_sim
                max_type = type
                max_item = item
    return max_sim, max_type, max_item



def _mean_embedding(embeddings):
    size = len(embeddings)
    return np.sum(embeddings, axis=0) / size

def _cal_cosine_sim(list_one, list_two):
    result = 1 - spatial.distance.cosine(list_one, list_two)
    return result


if __name__ == '__main__':
    max_sim, max_type, max_item = _find_max_sim(TEST, PRODUCT_MAP)
    print(max_sim)
    print(max_type)
    print(max_item)