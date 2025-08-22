from fastapi import FastAPI, File, UploadFile, Header, HTTPException
from fastapi.responses import JSONResponse
import cv2 as cv
import numpy as np
import os
import shutil
from typing import Optional
from uuid import uuid4

# ------------------- INIT -------------------

app = FastAPI()

# Create folders if not exist
os.makedirs("registered_faces", exist_ok=True)


# ------------------- HELPERS -------------------

def read_imagefile(file) -> np.ndarray:
    """Convert UploadFile to OpenCV image"""
    file_bytes = np.frombuffer(file, np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    return img


# ------------------- ENDPOINTS -------------------

@app.post("/register_face")
async def register_face(
        nic: str = Header(..., description="User ID"),
        image: UploadFile = File(...)
):
    """Register face → save image in user folder"""
    try:
        # Create folder
        user_folder = os.path.join("registered_faces", f"user_{nic}")
        os.makedirs(user_folder, exist_ok=True)

        # Load and save
        img = read_imagefile(await image.read())
        filename = f"{uuid4().hex}.jpg"
        filepath = os.path.join(user_folder, filename)
        cv.imwrite(filepath, img)

        return JSONResponse({
            "status": "success",
            "user_id": nic,
            "message": f"Image saved as {filename}",
            "file_path": filepath,
            "images_count": len(os.listdir(user_folder))
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete_user/{user_id}")
async def delete_user(user_id: str):
    """Delete user folder and all images"""
    try:
        user_folder = os.path.join("registered_faces", f"user_{user_id}")

        if not os.path.exists(user_folder):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        # Remove the entire user folder
        shutil.rmtree(user_folder)

        return JSONResponse({
            "status": "success",
            "user_id": user_id,
            "message": f"User folder and all images deleted successfully"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_users")
async def list_users():
    """List all registered users"""
    try:
        users = []
        registered_faces_dir = "registered_faces"

        if os.path.exists(registered_faces_dir):
            for item in os.listdir(registered_faces_dir):
                if item.startswith("user_"):
                    user_id = item.replace("user_", "")
                    user_folder = os.path.join(registered_faces_dir, item)
                    image_count = len([f for f in os.listdir(user_folder) if f.endswith('.jpg')])
                    users.append({
                        "user_id": user_id,
                        "image_count": image_count,
                        "folder_path": user_folder
                    })

        return JSONResponse({
            "status": "success",
            "users": users,
            "total_users": len(users)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user_images/{user_id}")
async def get_user_images(user_id: str):
    """Get list of images for a specific user"""
    try:
        user_folder = os.path.join("registered_faces", f"user_{user_id}")

        if not os.path.exists(user_folder):
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        images = [f for f in os.listdir(user_folder) if f.endswith('.jpg')]

        return JSONResponse({
            "status": "success",
            "user_id": user_id,
            "images": images,
            "total_images": len(images)
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return JSONResponse({
        "message": "Face Registration API",
        "endpoints": {
            "POST /register_face": "Register a face image for a user",
            "DELETE /delete_user/{user_id}": "Delete a user and all their images",
            "GET /list_users": "List all registered users",
            "GET /user_images/{user_id}": "Get images for a specific user"
        }
    })


# ------------------- RUN -------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)