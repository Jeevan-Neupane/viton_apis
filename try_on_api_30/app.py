from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gradio_client import Client, file
import cloudinary
import cloudinary.uploader
import os

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all HTTP headers
)

# Initialize Gradio Client
client = Client("Nymbo/Virtual-Try-On")

# Cloudinary Configuration
cloudinary.config(
    cloud_name="chatappjeevanneupane",
    api_key="241364295355246",
    api_secret="hPqBSyifpFh1pMwqe1hea_NEQW4"
)

# Define the request model for the input parameters
class TryOnRequest(BaseModel):
    background_url: str
    garm_img_url: str
    garment_des: str
    is_checked: bool
    is_checked_crop: bool
    denoise_steps: int
    seed: int

@app.post("/tryon")
async def try_on(request: TryOnRequest):
    try:
        # Use Gradio client to interact with the endpoint
        result = client.predict(
            dict={
                "background": file(request.background_url),
                "layers": [],
                "composite": None
            },
            garm_img=file(request.garm_img_url),
            garment_des=request.garment_des,
            is_checked=request.is_checked,
            is_checked_crop=request.is_checked_crop,
            denoise_steps=request.denoise_steps,
            seed=request.seed,
            api_name="/tryon"
        )
        
        # Cloudinary URLs to store
        cloudinary_urls = []
        
        # Upload each file from the result to Cloudinary
        for local_path in result:
            if os.path.exists(local_path):
                response = cloudinary.uploader.upload(
                    local_path,
                    folder="test",  # Specify folder in Cloudinary
                    use_filename=True,
                    unique_filename=True
                )
                cloudinary_urls.append(response['secure_url'])
        
        return {"cloudinary_urls": cloudinary_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
