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

pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
if(dataset_name == 'ted'): # for ted, the resolution is 384*384
    pixel = 384

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


def process_video(source_image, driving_video, fps):
    '''
    Preprocess
    '''
    source_image = resize(source_image, (pixel, pixel))[..., :3]
    driving_video = [resize(frame, (pixel, pixel))[..., :3] for frame in driving_video]

    '''
    Inference
    '''
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
    return [pred.tobytes() for pred in predictions], predictions[0].shape


class ImageResponse(BaseModel):
    image_bytes: bytes
    shape: tuple[int, int, int] = (256, 256, 3)
    
class VideoResponse(BaseModel):
    video: list[bytes]
    shape: tuple[int, int, int] = (256, 256, 3)

@app.post("/process_video")
# def process_video_route(image: UploadFile = File(...)):
def process_video_route(image: UploadFile = File(...), video: UploadFile = File(...)):
    # print(image.file.read())
    source_image = imageio.imread(io.BytesIO(image.file.read()))
    driving_video, fps = read_video(video)
    # reader = imageio.get_reader(io.BytesIO(video.read()))


    print("image received")

    predictions, output_shape = process_video(source_image, driving_video, fps)

    # return b'OK'
    return VideoResponse(video=driving_video, shape=output_shape)

    # return VideoResponse(predictions)