import torch
from torchvision import transforms
import time
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse, JSONResponse
import shutil
from pathlib import Path
import os
from typing import List
import cv2
import numpy as np
import pickle
from utilities import csv_converter, pose_to_num, get_pose_from_num, most_frequent, keypoints_parser, get_coords_line
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import tempfile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import requests

import supervision as sv
from ultralytics import YOLO
from object_gender_age import detect_age_gender, detect_faces, initialize_age_gender_models, category_dict, model_yolov10, MODEL_MEAN_VALUES, ageList, genderList, process_video_object
from pose_estimator_yolo import load_models, run_inference, draw_keypoints, process_video_frames

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
NN = None

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.on_event("startup")
async def startup_event():
    load_models()

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse(content={
            "filename": file.filename,
            "status": "success",
            "message": "Video uploaded successfully"
        })
    
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/pose_video/{filename}")
async def pose_video(
    filename: str,
    forcus: list[str] = Query(default=["fall"], description="List of forcus 'fall', 'fallen'"),
    warning: list[str] = Query(default=["fallen"], description="List of warning"),
    time_warning: int = Query(default=10, description="Time interval in seconds to trigger warnings")
):
    video_path = UPLOAD_DIR / filename
    
    if not video_path.exists():
        return JSONResponse(content={
            "status": "error",
            "message": "Video file not found"
        }, status_code=404)

    return StreamingResponse(
        process_video_frames(str(video_path), forcus, warning, time_warning),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get("/pose_video_url/")
async def pose_video_url(
    url: str,
    forcus: list[str] = Query(default=["fall"], description="List of forcus 'fall', 'fallen'"),
    warning: list[str] = Query(default=["fallen"], description="List of warning"),
    time_warning: int = Query(default=10, description="Time interval in seconds to trigger warnings")
):

    return StreamingResponse(
        process_video_frames(url, forcus, warning, time_warning),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )


@app.get("/object_video/{filename}")
async def object_video(
    filename: str,
    ban_objects: list[str] = Query(default=["knife"], description="List of danger objects"),
    ban_ages: list[str] = Query(default=None, description="List of ages to ban (e.g., '(0-2)', '(4-6)', etc.)"),
    ban_genders: list[str] = Query(default=None, description="List of genders to ban (e.g., 'Male', 'Female')")
):
    video_path = UPLOAD_DIR / filename
    if not video_path.exists():
        return JSONResponse({
            "status": "error",
            "message": "Video not found"
        }, status_code=404)

    # Validate banned objects, ages, and genders
    valid_ban_objects = [obj for obj in ban_objects if obj in category_dict.values()]
    valid_ban_ages = [age for age in ban_ages if age in ageList] if ban_ages else []
    valid_ban_genders = [gender for gender in ban_genders if gender in genderList] if ban_genders else []
    
    # Set defaults if lists are empty
    if not valid_ban_objects:
        valid_ban_objects = ["knife"]
    
    return StreamingResponse(
        process_video_object(str(video_path), valid_ban_objects, valid_ban_ages, valid_ban_genders),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.get("/object_stream/")
async def object_stream(
    url: str,
    ban_objects: list[str] = Query(default=["knife"], description="List of danger objects"),
    ban_ages: list[str] = Query(default=None, description="List of ages to ban (e.g., '(0-2)', '(4-6)', etc.)"),
    ban_genders: list[str] = Query(default=None, description="List of genders to ban (e.g., 'Male', 'Female')")
):

    valid_ban_objects = [obj for obj in ban_objects if obj in category_dict.values()]
    valid_ban_ages = [age for age in ban_ages if age in ageList] if ban_ages else []
    valid_ban_genders = [gender for gender in ban_genders if gender in genderList] if ban_genders else []

    if not valid_ban_objects:
        valid_ban_objects = ["knife"]
        
    return StreamingResponse(
        process_video_object(url, valid_ban_objects, valid_ban_ages, valid_ban_genders),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)