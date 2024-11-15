from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
import cloudinary
import cloudinary.uploader
import logging
import asyncio

# Directly configure Cloudinary with your credentials
cloudinary.config(
    cloud_name="chatappjeevanneupane",  # Replace with your Cloudinary cloud name
    api_key="241364295355246",        # Replace with your Cloudinary API key
    api_secret="hPqBSyifpFh1pMwqe1hea_NEQW4"   # Replace with your Cloudinary API secret
)

# Initialize FastAPI and Gradio client
app = FastAPI()

# Configure CORS middleware
origins = [
    "http://localhost:5173",  # Frontend address (use your actual frontend URL)
    "http://192.168.254.9:5173",  # Another common localhost URL
    # Add other allowed origins as necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow requests from these origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

client = Client("levihsu/OOTDiffusion")

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define request models with validation for image URLs
class ProcessHDRequest(BaseModel):
    vton_img: HttpUrl  # Ensures URL validity
    garm_img: HttpUrl  # Ensures URL validity
    n_samples: int = 1
    n_steps: int = 20
    image_scale: float = 2
    seed: int = -1

class ProcessDCRequest(BaseModel):
    vton_img: HttpUrl  # Ensures URL validity
    garm_img: HttpUrl  # Ensures URL validity
    category: str
    n_samples: int = 1
    n_steps: int = 20
    image_scale: float = 2
    seed: int = -1

# Helper function to call Gradio client asynchronously
async def predict_async(*args, **kwargs):
    return await asyncio.to_thread(client.predict, *args, **kwargs)

# Define the /process_hd endpoint
@app.post("/process_hd")
async def process_hd(request: ProcessHDRequest):
    try:
        # Log request details
        logging.info(f"Processing HD request: {request}")
        
        # Call Gradio client asynchronously
        result = await predict_async(
            vton_img=handle_file(request.vton_img),
            garm_img=handle_file(request.garm_img),
            n_samples=request.n_samples,
            n_steps=request.n_steps,
            image_scale=request.image_scale,
            seed=request.seed,
            api_name="/process_hd"
        )
        
        # Upload the resulting image to Cloudinary and get the URL
        cloudinary_response = cloudinary.uploader.upload(result[0]['image'], folder="test")
        result_url = cloudinary_response['secure_url']
        
        return {"result": [{"image": result_url, "caption": None}]}
    
    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail="Invalid input provided")
    except Exception as e:
        logging.error(f"Error processing HD request: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")

# Define the /process_dc endpoint
@app.post("/process_dc")
async def process_dc(request: ProcessDCRequest):
    try:
        # Log request details
        logging.info(f"Processing DC request: {request}")
        
        # Call Gradio client asynchronously
        result = await predict_async(
            vton_img=handle_file(request.vton_img),
            garm_img=handle_file(request.garm_img),
            category=request.category,
            n_samples=request.n_samples,
            n_steps=request.n_steps,
            image_scale=request.image_scale,
            seed=request.seed,
            api_name="/process_dc"
        )
        
        # Upload the resulting image to Cloudinary and get the URL
        cloudinary_response = cloudinary.uploader.upload(result[0]['image'], folder="test")
        result_url = cloudinary_response['secure_url']
        
        return {"result": [{"image": result_url, "caption": None}]}
    
    except ValueError as e:
        logging.error(f"ValueError: {e}")
        raise HTTPException(status_code=400, detail="Invalid input provided")
    except Exception as e:
        logging.error(f"Error processing DC request: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request")
