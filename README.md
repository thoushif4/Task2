# Task2
# Install required libraries
!pip install requests beautifulsoup4 transformers faiss-cpu
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np

# Function to crawl and scrape content from a website
def crawl_and_scrape(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = ' '.join([p.text for p in soup.find_all('p')])
    return text

# Function to segment content into chunks
def segment_content(text, chunk_size=512):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to convert chunks into vector embeddings
def convert_to_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(chunks, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Function to store embeddings in a vector database
def store_embeddings(embeddings, metadata):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, metadata

# Example usage
url = 'https://example.com'
text = crawl_and_scrape(url)
chunks = segment_content(text)
embeddings = convert_to_embeddings(chunks)
metadata = [{'url': url, 'chunk': chunk} for chunk in chunks]
index, metadata = store_embeddings(embeddings, metadata)
# Function to convert user query into vector embeddings
def query_to_embeddings(query, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(query, return_tensors='pt')
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return query_embedding

# Function to perform similarity search in the vector database
def similarity_search(query_embedding, index, metadata, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    results = [metadata[i] for i in indices[0]]
    return results

# Example usage
query = "What is the main topic of the website?"
query_embedding = query_to_embeddings(query)
results = similarity_search(query_embedding, index, metadata)
from transformers import pipeline

# Function to generate response using LLM
def generate_response(query, results, model_name='gpt-3.5-turbo'):
    context = ' '.join([result['chunk'] for result in results])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    generator = pipeline('text-generation', model=model_name)
    response = generator(prompt, max_length=200, num_return_sequences=1)
    return response[0]['generated_text']

# Example usage
response = generate_response(query, results)
print(response)
