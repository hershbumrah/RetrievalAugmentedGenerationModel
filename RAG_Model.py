
pip install faiss-cpu
pip install pymupdf
pip install transformers
pip install pandas
import pandas as pd
import numpy as np
import faiss
import pymupdf
import torch
import pprint
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

#Handling CSV file via pandas
df = pd.read_parquet("hf://datasets/iamtarun/python_code_instructions_18k_alpaca/data/train-00000-of-00001-8b6e212f3e1ece96.parquet")


#Creating a reader for the pdf of speech database
page_path = '/kaggle/input/food-data/food_data.pdf'
fullpdf = pymupdf.open(page_path, filetype = 'pdf')
pgdata = []
for page in fullpdf:
    text = page.get_text("text")
    linesperpage = text.split('\n')
    pgdata.extend(linesperpage)

#Close the document
fullpdf.close()
print(f'Number of Lines: {len(pgdata)}', end='\n\n') #lines
print("Sample: FoodData Central")

#Accessing the tokenizer and model
mytoken = "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"
login(token = mytoken, add_to_git_credential=True)

print("test")

#Loading the tokenizer and model selected from online
model_name = 'Meta-Llama'
tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/llama-3/transformers/8b-hf/1")
model = AutoModelForCausalLM.from_pretrained("/kaggle/input/llama-3/transformers/8b-hf/1")
simtokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

print("test2")

#Creating functions to employ in operations later on
def get_embeddings(text):
    inputs = simtokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512) #Tokenizes inputs so model can operate with them
    outputs = model(**inputs) #Passes the unpacked inputs dict to the model, allowing for output generation
    return outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy() #Extracts embeddings from model outputs

def faiss_index(vectors):
    dim = vectors.shape[1] #Vectors returns a tuple of (numvectors, numdims), so the second index provides dimensions
    index = faiss.IndexFlatL2(dim) #Uses L2 Index to Calculate Similarity Between Vectors
    index.add(vectors) #This adds vectors to the faiss-index
    return index

#In this next function, "k" corresponds to the number of returns that occur
def search_index(query, index, k = 3):
    queryvector = get_embeddings(query).reshape(1, -1) #tokenizes text and runs through model to return NumPy array of embeddings as 2D array with 1 row. Faiss search algorithm requires each row to be query vector.
    distances, indices = index.search(queryvector, k) #index.search returns array of the distances between the query vector and the 3 nearest neighbor vectors in the index, and a second array of indices of nearest neighbor vectors in index.
    return distances, indices #Returning two arrays, one of distances and one of indices as explained above

#Our functions are completed and will allow for simple and organized programming, now let's apply these to our selected text and model.
vectors = np.array([get_embeddings(text) for text in pgdata]) #List of text embeddings is converted into NumPy array. 2D array is created and each row corresponds to text embedding from pgdata.
main_index = faiss_index(vectors) #Builds a Faiss Index of our given vectors

query = "Nutritional Facts" #Select a query
distances, indices = search_index(query, main_index, k=3)
print("Distances:", distances[0])
print("Indices:", indices[0])
#We are not going to currently employ an API, but this can be done in the future.

print("test3")

#Now we will assign our prompt token
PROMPT = """
You are a Food Scientist that is responsible for answering questions about different foods and their nutritional information. 
You will be provided some fields and their definitions as additional information related to your specific use case. ENSURE ACCURACY.

First rewrite the query or question that is given. Inlcude many more details in the query to enable better search

If the orginal provided sufficient information to respond to the question, make valid == True
If it did not make it False.


RESPOND IN THIS FORMAT:

{
    'response': <response>,
    'query_rewrite': <rewritten_query>,
    'valid': <bool>
}

The Following is Document Context and the query from the User:
"""

#Now we will assign the response prompt token
RESPONSE_PROMPT = """
You are a Food Scientist that is responsible for answering questions about different foods and their nutritional information. 
You will be provided some fields and their definitions as additional information related to your specific use case. ENSURE ACCURACY.


RESPOND IN THIS FORMAT:

{'response': <response>, 'valid': <bool>}

The Following is Document Context and the query from the User:
"""
#Model is used to generate a response
def gen_response(query, ccont = [], index = main_index, k=5, model = model, lines = pgdata):
    distances, indices = search_index(query, index, k)
    line_similarity = [lines[i] for i in indices[0]]
    full_prompt = query + " " + " ".join(ccont) + " " + " ".join(line_similarity)
    inputs = tokenizer(full_prompt, return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length = 512, num_return_sequences = 1, no_repeat_ngram_size = 2, early_stopping = True)
    #Decodes the generated response
    full_generated_response = tokenizer.decode(outputs[0], skip_special_tokens = True)
    return full_generated_response

#Testing the program
test_prompt = "Tell me about food expiration dates"
try:
    generated_content = gen_response(test_prompt, max_length=100)
    print(generated_content)
except Exception as e:
    print(f"An error occurred: {e}")

print("test4")


