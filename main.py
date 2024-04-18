from typing import Union
from elasticsearch import Elasticsearch
from openai import OpenAI
import spacy
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
nlp = spacy.load("en_core_web_sm")
import os
from fastapi import FastAPI
openai_api_key = os.getenv("OPENAIKEY")
print(openai_api_key)
app = FastAPI()
client = OpenAI(api_key=openai_api_key)
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

es=Elasticsearch('https://localhost:9200',
                 basic_auth=("elastic","t20R5uG1ubcolVY3IN5e"),
                 ca_certs='http_ca.crt')

def validate_query(query):
    # Check if the query is not empty
    if not query:
        return False

    # Perform basic validation
    doc = nlp(query)
    # Check if the query contains at least one noun or verb
    if not any(token.pos_ in ["NOUN", "VERB"] for token in doc):
        return False

    # Additional validation rules can be added based on specific requirements

    return True

class Result(BaseModel):
    message:str
    result:list



@app.get("/search/eucliden/",summary="Euclidean Search", description="Send search  query as query parameter.\n Validation Makes sure query has atleast one noun or very you can skip it by setting validation False\n limit is the number of results you want",
         response_model=Result)
async def read_item(q: str,limit:int=20,validatation:bool=True):
    try:
        if validatation and not validate_query(q) :
            return {'message':'Not a valid query','result':[]}
        vector_of_input_keyword = get_embedding(q)
        query = {
            "field": "descvector",
            "query_vector": vector_of_input_keyword,
            "k": limit,
            "num_candidates": 1200,
        }
        res = es.knn_search(
            index="finalecle",
            knn=query,
            source=["productname", "description", "tags", "categories", "sub_categories","weightage"],
        )

        # Extract relevant information from search results
        hits = [{"_id": hit["_id"], "_source": hit["_source"],"_score": hit["_score"]} for hit in res['hits']['hits']]
        print(hits[0]["_score"])
        message=   'Result not Found'  if hits[0]["_score"]<6.8 else "success"
        sorted_data = sorted(hits, key=lambda x: not x['_source'].get('weightage', False))
        return {"message": message, "result": sorted_data}
    except Exception as e:
        return {"messagae":str(e),"result":[]}




@app.get("/search/cosine/",summary="Cosine Search", description="Send search  query as query parameter.\n Validation Makes sure query has atleast one noun or very you can skip it by setting validation False",
          response_model=Result)
async def read_item(q: str,limit:int=20,validatation:bool=True):
    try:
        if validatation and not validate_query(q) :
            return {'message':'Not a valid query','result':[]}
        vector_of_input_keyword = get_embedding(q)
        query = {
            "field": "descvector",
            "query_vector": vector_of_input_keyword,
            "k": limit,
            "num_candidates": 1200,
        }
        res = es.knn_search(
            index="cosinefast",
            knn=query,
            source=["productname", "description", "tags", "categories", "sub_categories"],
        )

        # Extract relevant information from search results
        hits = [{"_id": hit["_id"], "_source": hit["_source"],"_score": hit["_score"]} for hit in res['hits']['hits']]
        message=   'Result not Found'  if hits[0]["_score"]<6.8 else "success"
        sorted_data = sorted(hits, key=lambda x: not x['_source'].get('weightage', False))

        return {"message": message, "result": sorted_data}
    except Exception as e:
        return {"messagae":str(e),"result":[]}