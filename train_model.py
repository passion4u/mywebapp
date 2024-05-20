import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import optimizers

# 画像のディレクトリ
path_usd = 'images/usd/'
path_euro = 'images/euro/'

# 画像を格納するリストの作成
img_usd = []
img_euro = []

# 画像サイズの設定
img_size = 100

# 各カテゴリの画像サイズを変換してリストに保存
for img_name in os.listdir(path_usd):
    img = cv2.imread(os.path.join(path_usd, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img_usd.append(img)

for img_name in os.listdir(path_euro):
    img = cv2.imread(os.path.join(path_euro, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img_euro.append(img)

# データとラベルの設定
X = np.array(img_usd + img_euro)
y = np.array([0] * len(img_usd) + [1] * len(img_euro))

# ランダムに並び替え
rand_index = np.random.permutation(len(X))
X = X[rand_index]
y = y[rand_index]

# データ分割
X_train = X[:int(len(X) * 0.8)]
y_train = y[:int(len(y) * 0.8)]
X_test = X[int(len(X) * 0.8):]
y_test = y[int(len(y) * 0.8):]

# one-hotエンコーディング
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# VGG16を読み込み、転移学習を行う
input_tensor = Input(shape=(img_size, img_size, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# モデルの定義
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(rate=0.5))
top_model.add(Dense(32, activation='relu'))
top_model.add(Dropout(rate=0.5))
top_model.add(Dense(2, activation='softmax'))

# VGG16と追加層の連結
model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# VGG16の特徴抽出を固定
for layer in model.layers[:15]:
    layer.trainable = False

# モデルのコンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
              metrics=['accuracy'])

# 訓練を実行
history = model.fit(X_train, y_train, batch_size=64, epochs=50, validation_data=(X_test, y_test))

# モデルの保存
model.save('model.h5')
