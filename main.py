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

warnings.filterwarnings("ignore")
app = FastAPI()

def process_video(source_image, reader):
    '''
    Variables
    '''
    device = torch.device('cuda:0')
    dataset_name = 'vox' # ['vox', 'taichi', 'ted', 'mgif']
    config_path = 'config/vox-256.yaml'
    checkpoint_path = 'checkpoints/vox.pth.tar'
    predict_mode = 'relative' # ['standard', 'relative', 'avd']

    pixel = 256 # for vox, taichi and mgif, the resolution is 256*256
    if(dataset_name == 'ted'): # for ted, the resolution is 384*384
        pixel = 384

    '''
    Preprocess
    '''
    # source_image = imageio.imread(source_image_path)
    # reader = imageio.get_reader(driving_video_path)

    source_image = resize(source_image, (pixel, pixel))[..., :3]

    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

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
    return [{"image_bytes": pred.tobytes(), "shape": pred.shape} for pred in predictions]
    # imageio.mimsave(
    #     output_video_path, 
    #     [img_as_ubyte(frame) for frame in predictions], 
    #     fps=fps
    # )

class ImageResponse(BaseModel):
    image_bytes: bytes
    shape: tuple[int, int, int] = (256, 256, 3)
    
class VideoResponse(BaseModel):
    video: list[ImageResponse]

@app.post("/process_video")
def process_video_route(image: UploadFile = File(...), video: UploadFile = File(...)):
    source_image = imageio.imread(io.BytesIO(image.read()))
    reader = imageio.get_reader(io.BytesIO(video.read()))

    predictions = process_video(source_image, reader)

    return VideoResponse(predictions)