__import__('pysqlite3')
import sys
import sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import json
import re
import numpy as np
import hashlib
import streamlit as st
from collections import Counter
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
from embedding import create_index

# Constants
CHROMA_DATA_PATH = 'chromadb_fact_checker/'
COLLECTION_NAME = "document_embeddings"
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]

# API Clients
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_KEY,
                                                         model_name="text-embedding-3-small")

OpenAIClient = openai.OpenAI(api_key = OPENAI_KEY)
collection = client.get_or_create_collection(
    name = COLLECTION_NAME,
    embedding_function = openai_ef,
    metadata = {"hnsw:space" :  "cosine"}
)

# Note: Ensure you have proper imports and that 'lock' is defined in the appropriate scope.
lock = Lock()

def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return response["data"][0]["embedding"]

def semantic_search(query):

    try:
        result = collection.query(query_texts=[query], n_results=1)
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
            response = OpenAIClient.chat.completions.create(
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
    create_streamlit_interface()
