import gensim
gensim.__version__


import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

#nltk.download('stopwords')
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


df = pd.read_csv("lawall.csv", encoding='utf-8')

df = df[:100]

print('전체 문서의 수 :',len(df))

df['17'] = df['17'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
df

okt = Okt()

tokenized_data = []
for sentence in tqdm(df['17']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    tokenized_data.append(stopwords_removed_sentence)

print(tokenized_data[:3])


# 리뷰 길이 분포 확인
print('리뷰의 최대 길이 :',max(len(l) for l in tokenized_data))
print('리뷰의 평균 길이 :',sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

from gensim.models import Word2Vec
model = Word2Vec(tokenized_data, window=5, min_count=5, workers=4, sg=0)

print('완성된 임베딩 매트릭스의 크기 확인 :', model.wv.vectors.shape)


df[:5]

import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
from PIL import Image
from io import BytesIO
from nltk.tokenize import RegexpTokenizer
import nltk
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Word2Vec 모델 초기화
word2vec_model = Word2Vec(vector_size=300, window=5, min_count=2, workers=-1)
word2vec_model.build_vocab(tokenized_data)

# 한국어 워드 임베딩 불러오기
#ko_word_vectors = KeyedVectors.load_word2vec_format('/content/my_home/MyDrive/Colab Notebooks/ko.bin')

# Word2Vec 모델에 한국어 워드 임베딩 추가
#word2vec_model.wv.add(ko_word_vectors)

# 나머지 훈련 코드
word2vec_model.train(tokenized_data, total_examples=word2vec_model.corpus_count, epochs=15)


word2vec_model.corpus_count

word2vec_model = Word2Vec(vector_size = 300, window=5, min_count = 2, workers = -1)
word2vec_model.build_vocab(tokenized_data)
#word2vec_model.intersect_word2vec_format('/content/my_home/MyDrive/Colab Notebooks/ko.bin', lockf=1.0, binary=True)
word2vec_model.train(tokenized_data, total_examples = word2vec_model.corpus_count, epochs = 15)

def get_document_vectors(document_list, word2vec_model):
    document_embedding_list = []

    # 각 문서에 대해서
    for line in document_list:
        doc2vec = None
        count = 0
        for word in line.split():
            if word in word2vec_model.wv.key_to_index:
                count += 1
                # 해당 문서에 있는 모든 단어들의 벡터값을 더한다.
                if doc2vec is None:
                    doc2vec = word2vec_model.wv[word]
                else:
                    doc2vec = doc2vec + word2vec_model.wv[word]

        if doc2vec is not None:
            # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠준다.
            doc2vec = doc2vec / count
            document_embedding_list.append(doc2vec)

    # 각 문서에 대한 문서 벡터 리스트를 리턴
    return document_embedding_list


document_embedding_list = get_document_vectors(df['17'],word2vec_model)

print('문서 벡터의 수 :',len(document_embedding_list))

cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)

cosine_similarities[0]

print('코사인 유사도 매트릭스의 크기 :',cosine_similarities.shape)

def recommendations(information):
    tokenized_input = okt.morphs(information, stem=True)
    input_vector = sum([model.wv[word] for word in tokenized_input if word in model.wv.key_to_index])
    similarities = cosine_similarity([input_vector], document_embedding_list)[0]
    top_indices = similarities.argsort()[-5:][::-1]
    recommendations = df.iloc[top_indices][['3', '4']].reset_index(drop=True)
    return recommendations

recommendations("태영호")