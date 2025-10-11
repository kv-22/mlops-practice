import io
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from src.models.Generator import Generator

model = None
NOISE_DIM = 128


# To only load model once before receiving requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    path = os.path.join(os.getcwd(), "models/model.pth")
    model = Generator(noise_dim=NOISE_DIM)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()

    yield


app = FastAPI(lifespan=lifespan)


@app.get("/get_image")
async def get_image():
    try:
        with torch.no_grad():
            sample_noise = torch.randn(1, NOISE_DIM)
            generated_image = model(sample_noise)
            img_byte_arr = process_image(generated_image)
            return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception:
        raise HTTPException(
            status_code=500, detail="Sorry something went wrong at our end :("
        )


def process_image(image):
    image = image[0]
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    image_np = image.cpu().numpy().transpose((1, 2, 0))

    # Convert to uint8 (0-255 range)
    image_np = (image_np * 255).astype(np.uint8)

    # Create PIL Image
    pil_image = Image.fromarray(image_np)

    # Save to bytes buffer (in-memory, not to disk)
    img_byte_arr = io.BytesIO()  # Create buffer like creating a file
    pil_image.save(
        img_byte_arr, format="PNG"
    )  # Write image to buffer like saving the image
    img_byte_arr.seek(0)  # To read from start

    return img_byte_arr
