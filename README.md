# Mask Removal (Python: Streamlit)

日本語の説明: https://github.com/rtorii/Mask-Removal/blob/main/README_ja.md

**Note:** I originally deployed the app on Google App Engine, but since it costs about $20 / month, the app is currently disabled.

**Description:**

This app removes a face mask from an image of a person wearing it. Technically, it replaces the bottom half of the face with the face of another person.

1. detects a face from an image using the pretrained Pytorch face detection model (MTCNN) from https://github.com/timesler/facenet-pytorch. 
2. replaces the bottom half of the face(face mask portion) with an image of a random person using GAN from [https://github.com/zsyzzsoft/co-mod-gan](https://github.com/KYM384/co-mod-gan-pytorch). 
3. displays the processed image.

https://user-images.githubusercontent.com/52717342/166864383-f79315c0-e774-43af-b199-736400c56187.mov

**How to use the app:**

1. Open the app and press the `START` button.
2. Allow the app to access the webcam if it ask for the permission. Then the app displays the image from the webcam. 
3. The app processes the image from the webcam and displays it.

| Home page |  
| ------ | 
| <img width="961" alt="Screen Shot 2022-06-10 at 19 39 25" src="https://user-images.githubusercontent.com/52717342/173052321-3db86f98-21fd-430a-a3bb-99e1fb712ee0.png"> |  

If the app displays an error like the one below, please refresh your browser. This error is displayed when the streamlit_webrtc which is used to load the webcam video cannot be loaded due to reason such as slow internet.

<img width="692" alt="Screen Shot 2022-06-10 at 16 16 03" src="https://user-images.githubusercontent.com/52717342/173049060-ba300862-782a-4e19-a965-abbcd7526a1e.png">

Note: To run the app locally, please download `co-mod-gan-ffhq-9-025000_net_G_ema.pth` file from https://maildluteducn-my.sharepoint.com/:u:/g/personal/zengyu_mail_dlut_edu_cn/Ee1YPJG2Y7NDnUjJBf-SipoBBSlbv8QfFy6K7lsiiiiFHg?download=1. Then place the file in the same directory as `app.py`.



Created on 05/04/22 by team UUU.
