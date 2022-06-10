# マスク除去 (Python)

**説明:**

マスクをしている人の画像からマスクを除去するプログラムです。厳密に言えば、顔の下半分を他人の顔画像に置き換えるWebアプリです。そのため、マスクをしているしていないにかかわらず、顔の下半分を他人の顔画像に置き換えます。アプリは、Google App Engineで公開しています。

アプリのリンク：https://mask-352905.an.r.appspot.com/

**アプリがすること：**

1. 学習済みのPytorch顔検出モデル (MTCNN) で、画像から顔を認識します。
    - モデルの説明のリンク：https://github.com/timesler/facenet-pytorch
3. 顔の下半分（マスクの部分）を、 GANの学習済みモデルを使用し、他人の顔画像に置き換えます。
    - モデルの説明のリンク：https://github.com/KYM384/co-mod-gan-pytorch
4. 処理した画像を表示します。

https://user-images.githubusercontent.com/52717342/166864383-f79315c0-e774-43af-b199-736400c56187.mov


**アプリの使い方:**
1. アプリを開き、`START`ボタンを押します。
2. もしカメラのアクセス許可を求められたら、許可してください。許可したら、ウェブカメラから取り込んだ画像がスクリーンに映ります。
3. ウェブカメラから取り込んだ画像を処理し、表示します。

| ホームページ |  
| ------ | 
| <img width="961" alt="Screen Shot 2022-06-10 at 19 39 25" src="https://user-images.githubusercontent.com/52717342/173052321-3db86f98-21fd-430a-a3bb-99e1fb712ee0.png"> |  

もし下記のようなエラーが出た場合、ブラウザーをリフレッシュしてください。このエラーはネットワーク速度などの理由で、ウェブカメラの映像を読み込む際に使用するstreamlit_webrtcをローディングできないときに表示されます。

<img width="692" alt="Screen Shot 2022-06-10 at 16 16 03" src="https://user-images.githubusercontent.com/52717342/173049060-ba300862-782a-4e19-a965-abbcd7526a1e.png">

**アプリをローカルで実行する際の注意点：**

アプリを実行する前に、GANの学習済みモデル（`co-mod-gan-ffhq-9-025000_net_G_ema.pth`）を[こちら](https://maildluteducn-my.sharepoint.com/:u:/g/personal/zengyu_mail_dlut_edu_cn/Ee1YPJG2Y7NDnUjJBf-SipoBBSlbv8QfFy6K7lsiiiiFHg?download=1)からダウンロードし、`app.py`と同じディレクトリーに保存してください。

05/04/22にチームUUUによって作成。
