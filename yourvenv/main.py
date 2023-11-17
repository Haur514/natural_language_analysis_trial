import os
import zipfile

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# データの読み込み
train_images_path = "./content/train_images/"
label_master_path = "./content/label_master.tsv"
train_master_path = "./content/train_master.tsv"

label_master = pd.read_csv(label_master_path, delimiter='\t')
train_master = pd.read_csv(train_master_path, delimiter='\t')

# 学習データの読み込みと前処理
train_data = []
train_labels = []

# 前半2500枚を学習に使用
for index, row in train_master.head(2500).iterrows():
    image_path = os.path.join(train_images_path, row["file_name"])
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(96, 96))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # ピクセル値を正規化
    train_data.append(image)
    train_labels.append(row["label_id"])

train_data = np.array(train_data)
train_labels = np.array(train_labels)

# ラベルをone-hotエンコード
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=len(label_master))

# モデルの作成 (EfficientNetB0)
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(96, 96, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(len(label_master), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=predictions)

# モデルのコンパイル
model.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

# 学習率を動的に調整
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# 学習データと検証データに分割
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# データ拡張
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# モデルの学習
model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=15, callbacks=[reduce_lr])

# テストデータの読み込みと前処理
test_data = []
test_labels = []

# 後半500枚をテストに使用
for index, row in train_master.tail(500).iterrows():
    image_path = os.path.join(train_images_path, row["file_name"])
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(96, 96))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0  # ピクセル値を正規化
    test_data.append(image)
    test_labels.append(row["label_id"])

test_data = np.array(test_data)
test_labels = np.array(test_labels)

# テストデータの予測
predictions = model.predict(test_data)
predicted_labels = np.argmax(predictions, axis=1)

# 正解データ
true_labels = test_labels

# 予測と正解を結合
results = pd.DataFrame({
    "file_name": train_master.tail(500)["file_name"].values,
    "predicted_label_id": predicted_labels,
    "predicted_label": label_master["label_name"].iloc[predicted_labels].values,
    "true_label_id": true_labels,
    "true_label": label_master["label_name"].iloc[true_labels].values
})

# 正誤判定
results["correct"] = results["predicted_label_id"] == results["true_label_id"]

# TSVファイルへの出力
results.to_csv("./content/predicted_labels_and_results.tsv", sep='\t', index=False)

# 正答率の計算
accuracy = results["correct"].mean()
print(f"正答率: {accuracy * 100:.2f}%")
