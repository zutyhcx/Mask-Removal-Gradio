import cv2
import numpy as np
import torch
# import streamlit as st
from models.networks.co_mod_gan import Generator
from PIL import Image
import torchvision.transforms as tf
from facenet_pytorch import MTCNN
# from streamlit_webrtc import webrtc_streamer
import av

# st.title("Mask Removal")
# st.write("Note: This app accesses the webcam.")
# st.write("Description: The app removes a face mask from an image of a person wearing it. To start using it, press the START button.")

device="cpu"

class Options:
    mixing = 0
    load_size = 512
    crop_size = 512
    z_dim = 512
    model = "comod"
    netG = "comodgan"
    dlatent_size = 512
    num_channels = 3
    fmap_base = 16 << 10
    fmap_decay = 1.0
    fmap_min = 1
    fmap_max = 512
    randomize_noise = True
    architecture = 'skip'
    nonlinearity = 'lrelu'
    resample_kernel = [1,3,3,1]
    fused_modconv = True
    pix2pix = False
    dropout_rate = 0.5
    cond_mod = True
    style_mod = True
    noise_injection = True

opt = Options()


# @st.cache(allow_output_mutation=True)
def get_model():
    return torch.load("co-mod-gan-ffhq-9-025000.pth", map_location="cpu")
    # return torch.utils.model_zoo.load_url('https://streamlithonyakudata.s3.ap-northeast-1.amazonaws.com/temp.pth', model_dir=None, map_location="cpu")


netG_ema = Generator(opt)


# reading_model = st.empty()
# reading_model.text("Please wait while app loads the pretrained model...")
# netG_ema.load_state_dict(torch.load("temp.pth", map_location="cpu"))
netG_ema.load_state_dict(get_model())
netG_ema.to(device).eval()
# reading_model.empty()
# write_detect = st.empty()


transform = tf.Compose([
    tf.ToTensor(),
    tf.Normalize(0.5, 0.5)
])

z = torch.randn(1, 512, device=device)


@torch.no_grad()
def generate(image, mask, truncation):
    image = Image.fromarray(image)
    mask = Image.fromarray(mask).convert("L")
    x = transform(image).unsqueeze(0).to(device)
    mask = tf.ToTensor()(mask).unsqueeze(0).to(device)
    fake, _, _ = netG_ema(x, mask, [z])
    fake = fake[0].permute(1, 2, 0)
    fake = torch.clamp(fake, -1, 1)
    fake = (fake + 1) * 127.5
    fake = fake.to("cpu").numpy().astype(np.uint8)
    return fake

def get_mask(W, H, scale=1):
    coord = np.stack([
                np.tile(np.array(range(H)).reshape(H, 1), (1, W)),
                np.tile(np.array(range(W)).reshape(1, W), (H, 1)),
            ], axis=2).astype(np.float32)
    coord[:,:,0] = (coord[:,:,0] - H/2) / (H/2)
    coord[:,:,1] = (coord[:,:,1] - W/2) / (W/2)
    coord = np.sum(coord**2, 2) ** 0.5
    coord = 1 - coord
    coord *= scale
    coord[coord>1] = 1
    return np.tile(coord.reshape(H, W, 1), (1,1,3))


class FaceCrop:
    def __init__(self):
        self.detector = MTCNN(device=device)
    def face_detect(self, img):
        boxes, _, points = self.detector.detect(img, landmarks=True)
        if boxes is None:
            return []
        H, W, _ = img.shape
        x0, y0, x1, y1 = boxes[0]
        face = {}
        face["x1"] = x0
        face["y1"] = y0
        face["x2"] = x1
        face["y2"] = y1
        face["e0"] = points[0,0]
        face["e1"] = points[0,1]
        face["m"] = (points[0,3] + points[0,4]) / 2
        return face

    def face_align(self, img):
        lms = self.face_detect(img)
        H, W, _ = img.shape
        if len(lms) == 0:
            return []
        face = {}
        e0 = lms["e0"]
        e1 = lms["e1"]
        m = lms["m"]
        face["box"] = (lms["x1"], lms["y1"], lms["x2"], lms["y2"])
        x_tmp = e1 - e0
        y_tmp = (e0 + e1) / 2 - m
        c = ((e0 + e1) / 2 - 0.1 * y_tmp).astype(np.int32)
        s = int(max(4.0 * np.linalg.norm(x_tmp), 3.6 * np.linalg.norm(y_tmp)))
        r = int(s / 2**0.5)
        img_pad = np.pad(img, [[H//2, H//2], [W//2, W//2], [0, 0]], mode="reflect")
        c = (int(c[0] + W/2), int(c[1] + H/2))
        theta = np.arctan((e1[1]-e0[1])/(e1[0]-e0[0])) * 180 / np.pi
        M = cv2.getRotationMatrix2D(c, theta, 1)
        rotated = cv2.warpAffine(img_pad, M, (W*2, H*2))
        face["image"] = rotated[c[1]-s//2:c[1]-s//2+s, c[0]-s//2:c[0]-s//2+s]
        face["matrix"] = {
            "center" : c,
            "theta" : theta
        }
        return face

    def face_inverse(self, face_img, full_img, matrix):
        c_x, c_y = matrix["center"]
        theta = matrix["theta"]
        M = cv2.getRotationMatrix2D((c_x, c_y), -theta, 1)
        H, W, _ = full_img.shape
        h, w, _ = face_img.shape
        face_img = np.pad(face_img, [[c_y-h//2, H*2-(c_y+h//2)], [c_x-w//2, W*2-(c_x+w//2)], [0, 0]], mode="constant")
        face_mask = get_mask(w, h, scale=3)
        face_mask = np.pad(face_mask, [[c_y-h//2, H*2-(c_y+h//2)], [c_x-w//2, W*2-(c_x+w//2)], [0, 0]], mode="constant")
        face_img = cv2.warpAffine(face_img, M, (W*2, H*2))
        face_mask = cv2.warpAffine(face_mask, M, (W*2, H*2))
        face_img = face_img[H//2:H//2+H, W//2:W//2+W]
        face_mask = face_mask[H//2:H//2+H, W//2:W//2+W]
        img = full_img * (1-face_mask) + face_img * face_mask
        img = np.array(img, dtype=np.uint8)
        return img

mask_img = np.zeros((512, 512, 3)).astype(np.uint8)
mask_img[256:] = 255
f = FaceCrop()

# class VideoProcessor:

# from PIL import Image

def run(input_img):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    out_img = np.array(input_img, dtype=np.uint8)
    face = f.face_align(out_img)
    if len(face) > 0:
        print('face detected')
        orig_face_img = face["image"]
        H, W, _ = orig_face_img.shape
        face_img = cv2.resize(orig_face_img, (512, 512))
        face_img = generate(face_img, mask_img, 0.7)
        face_img = cv2.resize(face_img, (W, H))
        out_img = f.face_inverse(face_img, out_img, face["matrix"])
    else:
        print('face not detected')
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    return out_img

import os

if not os.path.exists('output_photos'):
    os.mkdir('output_photos')

for pic in os.listdir('photos'):#for each picture in the 'photos' folder
    input_pic = cv2.imread('photos' + "/" + pic, 1) # loads the picture
    if input_pic is None:
        continue
    output_pic = run(input_pic)
    output_pic_name = 'output_photos/'+pic
    cv2.imwrite(output_pic_name,output_pic)

# run(input_img)


# def recv(self,frame):
    # out_img = frame.to_ndarray(format="rgb24")
    # face = f.face_align(out_img)
    # if len(face) > 0: #もし画像から顔を感知した場合：顔の切り抜き画像をout_imgとして保存する。
    #     orig_face_img = face["image"]
    #     H, W, _ = orig_face_img.shape
    #     face_img = cv2.resize(orig_face_img, (512, 512))
    #     face_img = generate(face_img, mask_img, 0.7)
    #     face_img = cv2.resize(face_img, (W, H))
    #     out_img = f.face_inverse(face_img, out_img, face["matrix"])
    # out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    # return out_img

    # av.VideoFrame.from_ndarray(out_img, format="bgr24")

# webrtc_streamer(
#     key="example",
#     video_processor_factory=VideoProcessor,
#     media_stream_constraints={"video": True, "audio": False},
#     rtc_configuration={
#         "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     }
# )
