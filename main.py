import torch
import os

import imageio
import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings

from demo import load_checkpoints, make_animation
from skimage import img_as_ubyte

import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import cv2

warnings.filterwarnings("ignore")
app = FastAPI()

'''
Variables
'''
device = torch.device('cuda:0')
dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
config_path = 'config/vox-256.yaml'
checkpoint_path = 'checkpoints/vox.pth.tar'
predict_mode = 'relative' # ['standard', 'relative', 'avd']
output_video_path = "test.mp4"

# pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
# if(dataset_name == 'ted'): # for ted, the resolution is 384*384
#     pixel = 384

def read_video(videoUploadFile: UploadFile):
    tfile = tempfile.NamedTemporaryFile()
    tfile.write(videoUploadFile.file.read())
    vf = cv2.VideoCapture(tfile.name)
    fps = vf.get(cv2.CAP_PROP_FPS)

    cv_frames = []
    while True:
        ret, frame = vf.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv_frames.append(frame)

    return cv_frames, fps


def process_video(source_image, driving_video, fps, pixel):
    source_image = resize(source_image, (pixel, pixel))[..., :3]
    driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]


    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
        config_path = config_path, 
        checkpoint_path = checkpoint_path, 
        device = device
    )

    predictions = make_animation(
        source_image, 
        driving_video, 
        inpainting, 
        kp_detector, 
        dense_motion_network, 
        avd_network, 
        device = device, 
        mode = predict_mode
    )
    imageio.mimsave(output_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    predictions = np.array(predictions)
    return predictions.tobytes(), predictions.shape


@app.post("/process_video")
def process_video_route(image: UploadFile = File(...), video: UploadFile = File(...), size: int = 256):
    source_image = imageio.imread(io.BytesIO(image.file.read()))
    driving_video, fps = read_video(video)

    print("image received")

    predictions, output_shape = process_video(source_image, driving_video, fps, size)

    return FileResponse(output_video_path, media_type="video/mp4")

    # def generate(vid_bytes):
    #     yield from io.BytesIO(vid_bytes)

    # return StreamingResponse(generate(predictions), media_type="video/mp4")
