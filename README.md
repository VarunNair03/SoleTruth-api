This repository is the API component for [SoleTruth](https://huggingface.co/varunnair03/SoleTruth)

## Setup
Ensure that you have downloaded the pre-trained ViT model from the repo above.

Run the following command in the terminal to install the python dependencies
```shell
pip install -r requirements.txt
```

> **Note**: If you dont have a CUDA enabled gpu make sure you remove the index url from `requirements.txt`

I have changed the directory name of the fine-tuned model to `FineTuned-ViT-Model`. Ensure that you make the following changes in `app.py` or wherever the model is being called. Just pass in the name of your directory.

```python
from soletruth import SoleTruth

obj = SoleTruth("<PATH TO YOUR MODEL>")
```

Ensure you have the `.env` file present in your working directory.

It will look something like this:
```shell
QDRANT_CLUSTER = "<YOUR QDRANT CLUSTER URL>"
QDRANT_API_KEY = "<YOUR QDRANT CLUSTER API KEY>"
```

## Running the Code

To start the server you can either run `app.py` or run the following command in the terminal:
```shell
uvicorn app:app
```