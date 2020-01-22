import pickle, tfidf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import model_from_json


# 独自のテキストでテスト
text1 = """
好きなスケート選手は羽生選手です。
彼女はオリンピックで金メダルを取りました。
最近では野球やラグビーにもハマっています！
"""
text2 = """
常にAndroidのスマホを持っていて、Twitterばかり見ています。
このスマホに自作の簡単なアプリを入れて試験した事もあります。
ビンゴゲームやジャンケンゲームなどのアプリが当時の代表作です。
"""
text3 = """
幸せな結婚の秘訣は何でしょうか。
夫には敬意を、妻には愛情を示すことです。
そして子供を大切にすれば、円満な家庭が築けるでしょう。
"""
text4 = """
週末になるとアマゾンプライムやNetflixなどで、昔の作品を観ます。
特に、「雨に唄えば」「バック・トゥ・ザ・フューチャー」は
とても懐かしい気分になりました。実際に大きなスクリーンでも観てみたいです。
"""

# TF-IDFの辞書を読み込む
tfidf.load_dic("./text/genre-tdidf.dic")

# Kerasのモデルを定義&重みデータを読み込む
nb_classes = 4
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(29416,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy'])
model.load_weights('text/genre-model.hdf5')

# テキストを指定して判定
def check_genre(text):
    LABELS = ["スポーツ", "IT", "映画", "ライフ"]
    data = tfidf.calc_text(text)
    model._make_predict_function()
    pre = model.predict(np.array([data]))[0]
    n = pre.argmax()
    print(LABELS[n], "(", pre[n], ")")
    return LABELS[n], float(pre[n]), int(n) 

if __name__ == '__main__':
    check_genre(text1)
    check_genre(text2)
    check_genre(text3)
    check_genre(text4)

