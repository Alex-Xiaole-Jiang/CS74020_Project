# CS74020_Project
CS74020 Final Project -- Embedding Model

In this project we first aim to take an existing embedding model and fine-tune it with physics StackExchange Q&A.

The dataset is found on hugging face: https://huggingface.co/datasets/mteb/cqadupstack-physics

In particular, we fine tune with the SentenceTransformer package with the model "multi-qa-distilbert-cos-v1" with different loss function and different pooling method.

The my_pooling.py contains the pooling method, the utils.py contains a shuffling method, and there are several other ipynb files that does the data analysis.
