from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import cv2
import gdown
from ultralytics import YOLO

app = FastAPI()

# Allow CORS for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Drive file ID of your best.pt model (replace with your actual file ID)
GOOGLE_DRIVE_FILE_ID = "1fiIs9w4o5rP6eo-kP1qgHU8VsleiOj1F"
MODEL_PATH = "model/best.pt"

# Function to download model from Google Drive if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

# Download model if not available locally
download_model()

# Load YOLO model
model = YOLO(MODEL_PATH)  

# Create necessary directories if they don't exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/outputs", exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "YOLO detection app running"}

# Image detection endpoint
@app.post("/predict/image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Save uploaded image
        file_location = f"static/uploads/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run YOLO detection
        results = model(file_location)
        output_path = f"static/outputs/detected_{file.filename}"
        results[0].save(output_path)  # Access the first result and save it

        return JSONResponse(content={"message": "Detection completed", "output_path": output_path})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Video detection endpoint
@app.post("/predict/video/")
async def predict_video(file: UploadFile = File(...)):
    try:
        # Save uploaded video
        file_location = f"static/uploads/{file.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process video frame by frame
        cap = cv2.VideoCapture(file_location)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = f"static/outputs/detected_{file.filename}"
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO detection on each frame
            results = model(frame)
            annotated_frame = results[0].plot()  # Access the first result from the list
            out.write(annotated_frame)

            # Free memory every 30 frames
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames.")
                del results

        cap.release()
        out.release()

        return JSONResponse(content={"message": "Video detection completed", "output_path": output_path})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Endpoint to serve detected files
@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = f"static/outputs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename)
    return JSONResponse(status_code=404, content={"error": "File not found"})
