import torch
import os

import imageio
import imageio_ffmpeg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings


from skimage import img_as_ubyte

import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import tempfile
import cv2

output_video_path = "test.mp4"
app = FastAPI()

def read_video(path):
    vf = cv2.VideoCapture(path)
    fps = vf.get(cv2.CAP_PROP_FPS)

    cv_frames = []
    while True:
        ret, frame = vf.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv_frames.append(frame)

    return np.array(cv_frames).tobytes()


@app.get("/")
# def process_video_route(image: UploadFile = File(...)):
def process_video_route():
    print("request received")
    video = read_video("test.mp4")

    def generate(vid_bytes):
        yield from io.BytesIO(vid_bytes)
        
    return FileResponse(output_video_path, media_type="video/mp4")


    # return VideoResponse(predictions)