from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import torch
import os
from src.models.Generator import Generator
from PIL import Image
import io
import numpy as np
from contextlib import asynccontextmanager
from fastapi import HTTPException

model = None


# to only load model once before receiving requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    path = os.path.join(os.getcwd(), "models/model.pth")
    model = Generator(noise_dim=NOISE_DIM)
    model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()

    yield


app = FastAPI(lifespan=lifespan)

NOISE_DIM = 128


@app.get("/get_image")
async def get_image():
    try:
        with torch.no_grad():
            sample_noise = torch.randn(1, NOISE_DIM)
            generated_image = model(sample_noise)
            img_byte_arr = process_image(generated_image)
            return StreamingResponse(img_byte_arr, media_type="image/png")
    except Exception as e:
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
    img_byte_arr = io.BytesIO()  # create buffer like creating a file
    pil_image.save(
        img_byte_arr, format="PNG"
    )  # write image to buffer like saving the image
    img_byte_arr.seek(0)  # to read from start

    return img_byte_arr
