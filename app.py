# create a fastapi app with a single endpoint that returns a json response
# with the response being from soletruth.py as query_qdrant function
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from PIL import Image
import io
import os
from dotenv import load_dotenv
from soletruth import SoleTruth
import warnings
from pydantic import BaseModel
import json
warnings.filterwarnings("ignore")

class ImageData(BaseModel):
    content: dict

load_dotenv()
app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

soletruth_instance = SoleTruth()

@app.post("/query")
async def query_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Query Qdrant for similar images
        results = soletruth_instance.query_qdrant(image)

        # Return the results as a JSON response
        return JSONResponse(content=jsonable_encoder(results))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

@app.post("/upload") 
async def upload_image_with_metadata(
    image: UploadFile = File(Image),
    metadata: str = Form(...)
):
    # Parse metadata JSON string
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError:
        return JSONResponse(content={"error": "Invalid JSON in metadata"}, status_code=400)
    
    # insert the image and the metadata into the qdrant collection
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))

        # Get embeddings for the image
        # embeddings = soletruth_instance.get_embeddings(image)

        # Insert into Qdrant
        soletruth_instance.insert_into_vectordb(
            image= image,
            meta_data=metadata_dict,
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    

    return JSONResponse(content={
        "message": "Upload successful",
        "data": metadata_dict
    }, status_code=200)
    
# star t the server with uvicorn

if __name__ == "__main__":
    uvicorn.run(app=app, port=8000)




