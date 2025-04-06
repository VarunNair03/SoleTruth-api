import os
from PIL import Image
import torch
from transformers import (
    ViTImageProcessor ,
    ViTModel,
)
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import warnings
import random
warnings.filterwarnings("ignore")


class SoleTruth:
    def __init__(self, model_name:str = r"FineTuned-ViT-Model"):
        load_dotenv()
        self.model_name = model_name
        self.__initialize_qdrant()
        self.__load_model()

    def __initialize_qdrant(self) -> None:
        try:
            self.__qdrant = QdrantClient(
                url=os.getenv("QDRANT_CLUSTER"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            self.__qdrant_collection_name = "shoeprints_part1"
            print("Qdrant initialized successfully.")
        except Exception as e:
            print(f"Error initializing Qdrant: {e}")
            raise
    def __load_model(self) -> None:
        self.image_processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.model = ViTModel.from_pretrained(self.model_name)
        print("Model loaded successfully.")
        self.model.eval()

    def get_embeddings(self, image: Image.Image) -> list:
        """
        Get the embeddings for a given image using the ViT model.
        """
        try:
            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy().tolist()
            return embeddings
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            raise

    def query_qdrant(self, image: Image.Image, limit: int = 10) -> list:
        """
        Query Qdrant for similar images based on the embeddings.
        """
        try:
            torch.cuda.empty_cache()
            embeddings = self.get_embeddings(image)
            results = self.__qdrant.search(
                collection_name=self.__qdrant_collection_name,
                query_vector=embeddings[0],
                limit=limit,
            )
            torch.cuda.empty_cache()


            return results
        except Exception as e:
            print(f"Error querying Qdrant: {e}")
            raise

    def insert_into_vectordb(self, image: Image.Image, meta_data: dict) -> None:
        """
        Upload an image to Qdrant.
        """
        try:
            embeddings = self.get_embeddings(image)
            random_id = random.randint(1, 1_000_000)  # Generate a random integer ID
            self.__qdrant.upsert(
                collection_name=self.__qdrant_collection_name,
                points=[{
                    # use the generated random id for the image
                    "id": random_id,
                    "vector": embeddings[0],
                    "payload": {
                        "ID": meta_data["ID"],
                        "Gender": meta_data["Gender"],
                        "Brand": meta_data["Brand"], 
                        "Model/Details": meta_data["Model/Details"],
                        "Size": meta_data["Size"], 
                        "image_url": meta_data["image_url"],
                    }
                }]
            )
            print(f"Image uploaded successfully.")
        except Exception as e:
            print(f"Error uploading image: {e}")
            raise

if __name__ == "__main__":
    # Example usage
    image_path = r"patch_2.jpg"
    image = Image.open(image_path).convert("RGB")
    model = SoleTruth()
    results = model.query_qdrant(image, limit=10)
    print(results)




        
