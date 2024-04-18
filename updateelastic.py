from elasticsearch import Elasticsearch
import json
import pandas as pd
from pprint import pprint
import os
import numpy as np

from dotenv import load_dotenv
load_dotenv()
elasticpw = os.getenv("OPENAIKEY")
openai_api_key = os.getenv("OPENAIKEY")

from openai import OpenAI
client = OpenAI(api_key=openai_api_key)
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

es=Elasticsearch('https://localhost:9200',
                 basic_auth=("elastic",elasticpw),
                 ca_certs='http_ca.crt')

index_mapping = {
    "properties": {
        "description": {
            "type": "text"
        },
        "descvector": {
            "type": "dense_vector",
            "dims": 1536,
            "index": True,
            "similarity": "l2_norm",
            
        },
        "categories": {
            "type": "keyword"
        },
        "productname": {
            "type": "keyword"
        },
        "sub_categories": {
            "type": "keyword"
        },
        "tags": {
            "type": "keyword"
        }
    }
}
es.indices.create(index="euclidean", mappings=index_mapping)
embedding_df = pd.read_csv("embedded.csv",index_col=0)
embedding_df['descvector'] = embedding_df.descvector.apply(eval).apply(np.array)
docs = embedding_df.to_dict("records")

for doc in docs:
    try:
        es.index(index="euclidean", document=doc)
    except Exception as e:
        print(e)
