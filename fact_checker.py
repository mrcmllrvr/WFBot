import json
import re
import numpy as np
import torch
import hashlib
import streamlit as st
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
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

# Setup and downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')

# Constants
CHROMA_DATA_PATH = 'chromadb_fact_checker/'
COLLECTION_NAME = "document_embeddings"
openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]

# API Clients
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai.api_key, model_name="text-embedding-3-small")
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    if collection is None:
        collection = client.create_collection(name=COLLECTION_NAME, embedding_function=openai_ef, metadata={"hnsw:space": "cosine"})
except Exception as e:
    logging.error(f"Failed to create or retrieve the collection: {str(e)}")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
stop_words = set(stopwords.words('english'))

# Functions
def get_embedding(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return model_output.last_hidden_state.mean(dim=1).squeeze().tolist()

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
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
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

# Note: Ensure you have proper imports and that 'lock' is defined in the appropriate scope.
lock = Lock()



def create_index(file_path):
    """Load JSON data into a searchable index, where each entry is stored in chromadb for easy lookup and retrieval."""

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            if 'title' in entry and 'text' in entry:
                key = entry['title'].lower()
                content = entry['text']
                embedding = get_embedding(content)  # Generate the embedding for the document

                # Store the document and its embedding in ChromaDB
                try:
                    collection.add(ids=[key], embeddings=[embedding], metadatas=[{"title": key, "text": content}])
                except Exception as e:
                    print(f"Failed to add data into ChromaDB: {e}")

    print("Data indexing complete.")

def semantic_search(query):
    query_embedding = get_embedding(query)
    print("Query embedding:", query_embedding)  # Debug output

    try:
        result = collection.query(query_embeddings=[query_embedding], n_results=1)
        print(f"Queried results: {result}")
        
        # Checking and handling nested list structures in the query result
        if result and result.get('metadatas') and len(result['metadatas'][0]) > 0:
            similar_fact = result['metadatas'][0][0]  # Access the first list, then the first dictionary
            if 'text' in similar_fact:
                return similar_fact['text']
            else:
                return "Similar fact found, but no text available."
        else:
            return "No matching documents found. Please refine your query and try again."
    except Exception as e:
        print(f"Error querying chromadb: {e}")
        import traceback
        traceback.print_exc()  # Print full stack trace for debugging

    return "Oops, something went wrong with the query. Please try again later."

def check_fact(query):
    """Check if the user's query can be verified semantically with the data."""
    system_prompt = """
        Role: As a proficient educational assistant dedicated to supporting learners, your primary responsibilities include providing targeted feedback based on the WikiShia database. You should focus on enhancing student understanding without summarizing or deviating from the source material.

            Tasks:
            1. Critical Analysis and Feedback:
            - Assess each student's response individually to gauge their understanding of key concepts, using the WikiShia JSON database.
            - Provide concise, direct feedback to confirm, correct, or enhance understanding, based solely on information from the WikiShia JSON database.
            - Ensure feedback directly reflects the terminology and explanations as provided in the WikiShia database, strictly avoiding the introduction of general knowledge or interpretations not found in the database.
            - Use simple, clear language to maintain a supportive and educational tone.

            Handling Inquiries:
            1. For critiquing responses:
            - Offer direct feedback using only the information from the WikiShia JSON database. Avoid summarizing assessments.
            - Provide concise additional explanations to enhance clarity or address missing details, using only the WikiShia JSON database.
            - Directly correct inaccuracies and guide students back to relevant concepts from the WikiShia JSON content, particularly when responses are off-topic or incorrect.
            - Employ guided questions and provide additional information from the WikiShia content as necessary for follow-up queries or corrections.

            Response Guidelines:
            1. Ensure all feedback is accurate and exclusively supported by the WikiShia JSON database.
            2. Provide corrective guidance and additional information if responses misinterpret a concept, using only the WikiShia JSON content.
            3. Use concise questions and dialogue to encourage critical thinking, adhering strictly to WikiShia content.
            4. Maintain a supportive and educational tone, using simple language and practical examples drawn exclusively from the WikiShia JSON database.
            5. Aim for engagement through direct and educational feedback, strictly adhering to the WikiShia content without summarizing or providing extraneous details.
            """


    matched_content = semantic_search(query)

    if matched_content not in ["No similar documents found.", "Found a document but it's not similar enough."]:
        try:
            response = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": matched_content}
                ]
            )
            if response.choices:
                message_content = response.choices[0].message.content  # Access the content correctly
                return message_content.strip() if message_content else "No response generated."
            else:
                return "No choices available in response."
        except Exception as e:
            # Providing a detailed error message in case of an API or other exception
            return f"An error occurred: {str(e)}"
    else:
        return "Unable to find a matching document or the document is not similar enough."

def create_streamlit_interface():
    
    # Using columns to center an image
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.empty()

    with col2:
        st.image('logo.png', width=300)  

    with col3:
        st.empty()


    st.title('🤖 WikiShia Fact Checker Chatbot')
    st.write("This chatbot's here to help you dive into all things Shia Islam! Discover accurate info on various topics in a fun, easy way. It's like having a helpful friend guiding your learning journey. Let's explore and learn together!")

    # CSS for styling message history, fixed chat input, and labels
    st.markdown("""
        <style>
            .message-container, .message-chatbot {
                padding: 10px;
                margin-top: 5px;
                border-radius: 5px;
            }
            .message-container {
                background-color: #f0f0f0; /* Light grey background for User */
                border-left: 5px solid #4CAF50; /* Green border for User */
            }
            .message-chatbot {
                background-color: #ffffff; /* White background for Chatbot */
                border-left: 5px solid #2196F3; /* Blue border for Chatbot */
            }
            .fixed-footer {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background-color: #fff;
                padding: 10px 20px;
                box-shadow: 0px -2px 10px rgba(0,0,0,0.1);
                z-index: 100;
            }
            .streamlit-container {
                padding-bottom: 70px; /* Ensure padding for fixed footer input */
            }
            .label {
                font-weight: bold;
                display: block;
                margin-bottom: 5px;
            }
            .message-text {
                margin-left: 20px; /* Indent message text for better readability */
            }
            .thread-container {
                margin-top: 20px;
                margin-bottom: 20px;
            }
            .stButton > button {
                width: 100%;
                border-radius: 5px;
                background-color: #4CAF50;
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)

    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []

    def ask_question():
        user_query = st.session_state.query
        if user_query:
            # Append the user query immediately to the chat history
            st.session_state['message_history'].append({'sender': '👤User', 'text': user_query})
            
            # Show spinner while processing the response
            with st.spinner('Crafting response...'):
                response = check_fact(user_query)  # Adjusted to not use index or embeddings
                st.session_state['message_history'].append({'sender': '🤖Chatbot', 'text': response})

    def start_new_chat():
        st.session_state['message_history'] = []

    # # Display messages using HTML and CSS in a scrollable container
    for message in st.session_state['message_history']:
        if message['sender'] == '👤User':
            class_name = "message-container"
            label = "👤User"
        else:
            class_name = "message-chatbot"
            label = "🤖Chatbot"

        # Assume that paragraphs are split by '\n' in the message["text"]
        # This converts each line into a paragraph within <p> tags
        paragraphs = message["text"].split('\n')  # Splits the text into paragraphs at newlines
        paragraph_html = ''.join(f'<p>{paragraph}</p>' for paragraph in paragraphs if paragraph.strip())  # Wraps non-empty paragraphs in <p> tags

        st.markdown(f'''
        <div class="thread-container">
            <div class="{class_name}">
                <div class="label">{label}:</div>
                <div class="message-text">{paragraph_html}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)


    # Fixed footer for chat input
    st.markdown('<div class="fixed-footer">', unsafe_allow_html=True)
    st.chat_input("Type your query here:", key="query", on_submit=ask_question)
    st.markdown('</div>', unsafe_allow_html=True)

    # Button to start a new chat at the bottom of the conversation
    if st.button("Start New Chat", key='start_new_chat'):
        start_new_chat()


# Adjust the main part to work with session state
if __name__ == '__main__':
    json_file_path = "data/scraped-Beliefs.jsonl"
    create_index(json_file_path)
    create_streamlit_interface()
