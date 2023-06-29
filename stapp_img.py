### 事前準備 ###
#
# AiPro2022フォルダに移動して…
#
# > py  -3.9  -m  venv  .tfflow     ← 新しい仮想環境「.tfflow」を生成する
#
# > .tfflow\Scripts\activate.bat    ← 新しい仮想環境「.tfflow」を起動する
#
# (.tfflow) > python -m pip install -r requirements.txt   ← ライブラリの一括インストール
#
# ※一括インストールは時間がかかりますので(10分～？）休み時間の前などに実行するとよいでしょう
#
#
# requirements.txt の内容
#   streamlit             ← 毎度、おなじみ
#   typing_extensions     ← これがないとエラーが出ることがあるので一応
#   numpy                 ← 毎度、おなじみ
#   pandas                ← 毎度、おなじみ
#   tensorflow-cpu        ← 深層学習用ライブラリ（容量の関係で今回はCPU版を採用）
#
#
# (.tfflow) > streamlit run stapp_img.py    ← Webアプリ（stapp_img.py）を起動


### Teachable Machineでモデルを作成する方法（中級以上） ###
#
#（Teachable Machineの詳細は、「前期」の授業で実施した「認識編Ⅰ」を参照）
# 
# ※推奨ブラウザ：Google Chrome（他のブラウザだと、ファイルのD&Dができないかも）
# 
# 1. Teachable Machineにアクセスする → 使ってみる → 画像プロジェクト → 標準の～
#   https://teachablemachine.withgoogle.com/
#   
# 2. 各Classの「アップロード」ボタンからデータを追加する
# 
# 3. トレーニングを実行する
#
# 4. モデルをエクスポートする　→　Tensorflowタブ → Keras → モデルをダウンロード
#
# 5. モデルのファイル（converted_keras.zip）がダウンロードされる
# 
# 6. converted_keras.zip に含まれる「keras_model.h5」を、
#    この stapp_img.py ファイルと「同じフォルダ」に格納する
#
# 7. 準備OK！
# 
# （以後、モデルを作り直すたびに、エクスポート → .h5ファイルの上書きをしてください）


# ライブラリのインポート
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np


# 画像(img)が属するクラスを推論する関数（'weights_file'は、モデルのファイル名）
# 以下、Teachable Machineのエクスポート時に自動生成されるコードをコピペする(済)
def teachable_machine_classification(img, weights_file):

    # モデルの読み込み
    model = load_model(weights_file)

    # kerasモデルに投入するのに適した形状の配列を作成する。
    # 配列に入れることができる画像の「長さ」または枚数は
    # shapeタプルの最初の位置（この場合は1）で決まる。
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # これを画像へのパスに置き換える
    # image = Image.open(img)
    image = img

    # Teachable Machineと同じ方法で、224x224にリサイズする。
    # 少なくとも224x224になるように画像をリサイズし、中心から切り取る。
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # 画像をnumpyの配列に変換する
    image_array = np.asarray(image)

    # 画像の正規化
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # 画像を配列に読み込む
    data[0] = normalized_image_array

    # 推論を実行する
    prediction = model.predict(data)

    # 推論結果をメインモジュールに戻す
    return prediction.tolist()[0]


# メインモジュール
def main():

    # タイトルの表示
    st.title("城判別")

    # アップローダの作成
    uploaded_file = st.file_uploader("Choose a Image...", type="jpg")

    # 画像がアップロードされた場合...
    if uploaded_file is not None:

        # 画像を画面に表示
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # teachable_machine_classification関数に画像を引き渡してクラスを推論する
        prediction = teachable_machine_classification(image, 'keras_model.h5')        
        st.caption(f'推論結果：{prediction}番') # 戻り値の確認（デバッグ用）

        classNo = np.argmax(prediction)          # 一番確率の高いクラス番号を算出
        st.caption(f'判定結果：{classNo}番')      # 戻り値の確認（デバッグ用）

        # 推論の確率を小数点以下3桁で丸め×100(%に変換)
        pred0 = round(prediction[0],3) * 100  # 猫の確率(%)
        pred1 = round(prediction[1],3) * 100  # 犬の確率(%)

        # 推論で得られたクラス番号(初期値は0)によって出力結果を分岐
        if classNo == 0:
            st.subheader(f"これは{pred0}％の確率で「siro1」です！")
        else:
            st.subheader(f"これは{pred1}％の確率で「siro2」です！")




# mainの起動
if __name__ == "__main__":
    main()


#=============================================================================#
#
# 課題. 画像分類ツールの作成
#
# 初級（必須）：新しく仮想環境(.tfflow)を作成して、このプログラムを稼働させる
#
# 中級（任意）：「犬」「猫」「アザラシ」を認識できる新しいモデルを作成する
#
# 上級（任意）：画像認識を利用したオリジナルのWebアプリを作成しよう(自由課題)
#
#              アイデア1) マニアックな物を認識してくれる「◯◯鑑定士」アプリ
#              アイデア2) 撮影した物が、何の動物に似ているかを判定するアプリ
#
#              ※Streamlitの各種APIやコンポーネントなどは自由に使って結構です
#
# #-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-#
#
# 提出方法： 今回の stapp_img.py は提出の必要はありません。
#
#            次の授業で Streamlitアプリの「デプロイ（公開）」を実施しますので、
#            その際、このアプリをデプロイして取得したURLを提出していただきます
#
#=============================================================================#
