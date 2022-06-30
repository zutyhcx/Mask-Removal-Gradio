# マスク除去 (Python)





**説明:**

マスクをしている人の画像からマスクを除去するプログラムです。厳密に言えば、顔の下半分を他人の顔画像に置き換えるWebアプリです。そのため、マスクをしているしていないにかかわらず、顔の下半分を他人の顔画像に置き換えます。

Streamlitを使用し、Webアプリ化しました。その後Streamlit CloudとHerokuにアプリをデプロイすることを試しましたが、モデルとインストールするライブラリ（特にPyTorch）の容量が大きいためできませんでした。Google App Engineにデプロイできましたが、１ヶ月に2000円ほどかかる為、現在はアプリをデプロイしていません。

**プログラムの手順：** https://scrumsign.com/untitled

https://user-images.githubusercontent.com/52717342/166864383-f79315c0-e774-43af-b199-736400c56187.mov


**プログラムをローカルで実行する際の注意点：**

プログラムを実行する前に、GANの学習済みモデル（`co-mod-gan-ffhq-9-025000_net_G_ema.pth`）を[こちら](https://maildluteducn-my.sharepoint.com/:u:/g/personal/zengyu_mail_dlut_edu_cn/Ee1YPJG2Y7NDnUjJBf-SipoBBSlbv8QfFy6K7lsiiiiFHg?download=1)からダウンロードし、`app.py`と同じディレクトリーに保存してください。

**アプリの使い方:**
1. アプリを開き、`START`ボタンを押します。
2. もしカメラのアクセス許可を求められたら、許可してください。許可したら、ウェブカメラから取り込んだ画像がスクリーンに映ります。
3. ウェブカメラから取り込んだ画像を処理し、表示します。

| ホームページ |  
| ------ | 
| <img width="961" alt="Screen Shot 2022-06-10 at 19 39 25" src="https://user-images.githubusercontent.com/52717342/173052321-3db86f98-21fd-430a-a3bb-99e1fb712ee0.png"> |  

<!-- もし下記のようなエラーが出た場合、ブラウザーをリフレッシュしてください。このエラーはネットワーク速度などの理由で、ウェブカメラの映像を読み込む際に使用するstreamlit_webrtcをローディングできないときに表示されます。

<img width="692" alt="Screen Shot 2022-06-10 at 16 16 03" src="https://user-images.githubusercontent.com/52717342/173049060-ba300862-782a-4e19-a965-abbcd7526a1e.png"> -->

05/04/22にチームUUUによって作成。
