import json
import re
import numpy as np
import hashlib
import streamlit as st
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
import chromadb
from chromadb.utils import embedding_functions
import openai
import logging
import time
from threading import Lock

# Constants
CHROMA_DATA_PATH = 'chromadb_fact_checker/'
COLLECTION_NAME = "document_embeddings"
openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]

# API Clients
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key,
                                                         model_name="text-embedding-3-small")
collection = client.get_or_create_collection(
    name = COLLECTION_NAME,
    embedding_function = openai_ef,
    metadata = {"hnsw:space" :  "cosine"}
)

# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
stop_words = set(stopwords.words('english'))

# Functions
def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return response["data"][0]["embedding"]


    # encoded_input = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    # with torch.no_grad():
    #     model_output = model(**encoded_input)
    # return model_output.last_hidden_state.mean(dim=1).squeeze().tolist()

def preprocess_text(text, use_lemmatization=True):
    normalized_text = re.sub(r'\W+', ' ', text.lower())
    tokens = word_tokenize(normalized_text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    final_tokens = []
    lemmatizer = WordNetLemmatizer() if use_lemmatization else PorterStemmer()
    for token, tag in pos_tag(filtered_tokens):
        if use_lemmatization:
            final_tokens.append(lemmatizer.lemmatize(token))
        else:
            final_tokens.append(lemmatizer.stem(token))
    return final_tokens

def generate_unique_id(document):
    hash_object = hashlib.sha256(document.encode('utf-8'))
    return hash_object.hexdigest()


logging.basicConfig(level=logging.INFO)


lock = Lock()



class EnhancedKeywordExtractor:
    # Initialize the class attribute for tracking session start time
    session_start_time = time.time()
    
    def __init__(self, documents, client, collection):
        self.client = client
        self.collection = collection
        self.lock = Lock()

        # Check if more than a week has passed and refresh existing IDs if so
        if time.time() - EnhancedKeywordExtractor.session_start_time > 604800:  # 604800 seconds in a week
            self.refresh_existing_ids()
            EnhancedKeywordExtractor.session_start_time = time.time()  # Reset the session start time
        else:
            self.existing_ids = self.fetch_existing_ids()
        
        logging.info(f"Initial existing IDs: {self.existing_ids}")
        self.upsert_documents(documents)

    def fetch_existing_ids(self):
        """Fetch a set of all existing IDs from the database."""
        existing_ids = set()
        try:
            documents = self.collection.list_documents()  # Assumed method to fetch all documents
            existing_ids = {doc.id for doc in documents}
        except Exception as e:
            logging.error(f"Failed to fetch existing IDs: {str(e)}")
        return existing_ids

    def refresh_existing_ids(self):
        """Refresh the set of existing IDs from the database."""
        with self.lock:
            try:
                documents = self.collection.list_documents()
                self.existing_ids = {doc.id for doc in documents}
                logging.info("Existing IDs refreshed.")
            except Exception as e:
                logging.error(f"Failed to refresh existing IDs: {str(e)}")

    def add_or_update_document(self, document, embedding, metadata, doc_id):
        """Add or update a document in the database."""
        with self.lock:
            print(doc_id)
            # Ensure ID list is up to date before each operation
            if doc_id in self.existing_ids:
                self.update_document(document, embedding, metadata, doc_id)
            else:
                self.insert_new_document(document, embedding, metadata, doc_id)

    def update_document(self, document, embedding, metadata, doc_id):
        """Update an existing document in the database."""
        try:
            self.collection.update(documents=[document], embeddings=[embedding], metadatas=[metadata], ids=[doc_id])
            logging.info("Document updated successfully.")
        except Exception as e:
            logging.error(f"Error updating document {doc_id}: {str(e)}")

    def insert_new_document(self, document, embedding, metadata, doc_id):
        """Insert a new document into the database."""
        try:
            self.collection.add(documents=[document], embeddings=[embedding], metadatas=[metadata], ids=[doc_id])
            self.existing_ids.add(doc_id)
            logging.info("Document inserted successfully.")
        except Exception as e:
            logging.error(f"Error inserting document {doc_id}: {str(e)}")

    def upsert_document(self, document, embedding, metadata, doc_id):
        """Attempt to upsert a document into the database using transaction."""
        with self.lock:
            try:
                self.collection.begin_transaction()  # Begin a transaction
                existing = self.collection.get(id=doc_id)  # Check if the document exists
                if existing:
                    self.collection.update(documents=[document], embeddings=[embedding], metadatas=[metadata], ids=[doc_id])
                else:
                    self.collection.add(documents=[document], embeddings=[embedding], metadatas=[metadata], ids=[doc_id])
                self.collection.commit_transaction()  # Commit the transaction
                logging.info(f"Document with ID {doc_id} upserted successfully.")
            except Exception as e:
                self.collection.rollback_transaction()  # Rollback in case of an error
                logging.error(f"Error upserting document {doc_id}: {str(e)}")


def create_index(file_path):
    """Load JSON data into a searchable index, where each entry is stored in chromadb for easy lookup and retrieval."""
    for_chunking = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            if 'title' in entry and 'text' in entry:
                key = entry['title'].lower()
                content = entry['text']
                import os
                os.write(1,f"TOTAL LENGTH : {len(content)}\n".encode())
                try:
                    embedding = get_embedding(content)  # Generate the embedding for the document
                    collection.add(ids=[key], embeddings=[embedding], metadatas=[{"title": key, "text": content}])
                except:
                    os.write(1,f"{key} is too long. Consider chunking\n".encode())
                    for_chunking.append(key)

    print("Data indexing complete.")
    return for_chunking

if __name__ == '__main__':
    import os
    
    json_file_path = "data/scraped-Beliefs.jsonl"
    to_chunk = create_index(json_file_path)
    
    os.write(1,f"{len(to_chunk)} Documents to Chunk : {'|'.join(to_chunk)}".encode())




