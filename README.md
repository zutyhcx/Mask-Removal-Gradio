# Mask Removal (Python: Streamlit)

日本語の説明: https://github.com/rtorii/Mask-Removal/blob/main/README_ja.md

Link to the app: https://mask-352905.an.r.appspot.com/

**Description:**

This app removes a face mask from an image of a person wearing it. Technically, it replaces the bottom half of the person's face and replaces it with the face of another person.

1. detects a face from an image using the pretrained Pytorch face detection model (MTCNN) from https://github.com/timesler/facenet-pytorch. 
2. replaces the bottom half of the face(face mask portion) with an image of a random person using GAN from https://github.com/zsyzzsoft/co-mod-gan.
3. returns the image.

https://user-images.githubusercontent.com/52717342/166864383-f79315c0-e774-43af-b199-736400c56187.mov

Created on 05/04/22 by team UUU.
