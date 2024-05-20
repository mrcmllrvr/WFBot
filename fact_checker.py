__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3
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

# Constants
CHROMA_DATA_PATH = 'chromadb_fact_checker/'
COLLECTION_NAME = "document_embeddings"
OPENAI_KEY = st.secrets["OPENAI_API_KEY"]

# API Clients
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_KEY,
                                                         model_name="text-embedding-3-small")

# Initialize the Questions
topic_choices = {
  "Tawhid" : [
    "What is the literal meaning of the word Tawhid?",
    "What is the importance or place of Tawhid in Islam?",
    "What are the levels of Tawhid?",
  ],
  "Prophethood" : [
    "What was the goal of the Prophets?",
    "What is Prophethood?",
    "What is the necessity of Prophethood?",
  ],
  "Qiyama" : [
    "What are some of the signs of Qiyama mentioned in the Quran?",
    "What is Qiyama?",
    "Which issues will people be questioned about on Qiyama according to the Quran?",
  ]
}

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

def check_fact(bot_question, query):
    """Check if the user's query can be verified semantically with the data."""
    system_prompt = """
        Role: As a proficient educational assistant dedicated to supporting learners, your primary responsibilities include providing targeted feedback based solely on the information from the designated database. You should focus on enhancing student understanding without summarizing or deviating from the source material.

            Tasks:
            1. Critical Analysis and Feedback:
                - Assess each student's response individually to gauge their understanding of key concepts, using only the information from the designated database.
                - Provide concise, direct feedback to confirm, correct, or enhance understanding, based exclusively on information from the designated database.
                - Ensure feedback directly reflects the terminology and explanations from the designated database, strictly avoiding the introduction of general knowledge or interpretations not found in the database.
                - Use simple, clear language to maintain a supportive and educational tone.
            
            Handling Inquiries:
            1. For critiquing responses:
                - Offer direct feedback using only the information from the designated database. Avoid summarizing assessments.
                - Provide concise additional explanations to enhance clarity or address missing details, using only information from the designated database.
                - Directly correct inaccuracies and guide students back to relevant concepts from the designated database, particularly when responses are off-topic or incorrect.
                - Employ guided questions and provide additional information from the designated database as necessary for follow-up queries or corrections.
            
            Response Guidelines:
            1. Ensure all feedback is accurate and exclusively supported by the designated database.
            2. Provide corrective guidance and additional information if responses misinterpret a concept, using only the designated database.
            3. Use concise questions and dialogue to encourage critical thinking, adhering strictly to the designated database.
            4. Maintain a supportive and educational tone, using simple language and practical examples drawn exclusively from the designated database.
            5. Aim for engagement through direct and educational feedback, strictly adhering to the designated database without summarizing or providing extraneous details.
            6. Avoid explicitly mentioning the source of information; act as if the designated database is the inherent source of truth.
            7. In the instance that the student provides an incomprehensible answer, dont try to make sense of the answer - respond accordingly but restrict your responses to information you know about the supposed answer as based of the designated database.
            """


    matched_content = semantic_search(query)

    if matched_content not in ["No similar documents found.", "Found a document but it's not similar enough."]:
        try:
            response = OpenAIClient.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content" : bot_question},
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
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #002855;
        }
        .sidebar-img {
            width: 50%;
            height: auto;  /* Maintain aspect ratio */
            max-width: 150px;  /* Set a max width for the images */
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        st.sidebar.image("world-federation-logo.png", use_column_width=True)
        st.sidebar.image("ICAIR-logo.png", use_column_width=True)

    # Using columns to center an image
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.empty()

    with col2:
        st.image("WF-Chatbot-logo2.jpg", width=50)

    with col3:
        st.empty()

    st.title('Muallim Assistant')
    st.write("Hi! How can I help you today?")

    
    # Initialize Session States
    if 'current_question' not in st.session_state:
        st.session_state['current_question'] = None

    if 'question_choices' not in st.session_state:
        st.session_state['question_choices'] = None

    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []

    st.session_state.disabled = st.session_state['current_question'] is not None

    # Topic Selection
    def disable_buttons_and_topic_select(topic):
        st.session_state.disabled = True
        st.session_state['question_choices'] = topic_choices[topic]
        st.session_state['current_question'] = np.random.choice(st.session_state['question_choices'])
        st.session_state['message_history'].append({'sender': "🤖Chatbot", 'text': st.session_state['current_question']})
    question_col1, question_col2, question_col3 = st.columns(3)

    button1 = question_col1.button("Prophethood",
                                   key="q1_clicked",
                                   on_click = disable_buttons_and_topic_select,
                                   args = ['Prophethood'],
                                   disabled=st.session_state.disabled)
    button2 = question_col2.button("Tawhid",
                                   key="q2_clicked",
                                   on_click = disable_buttons_and_topic_select,
                                   args = ['Tawhid'],
                                   disabled=st.session_state.disabled)
    button3 = question_col3.button("Qiyama",
                                   key="q3_clicked",
                                   on_click = disable_buttons_and_topic_select,
                                   args = ['Qiyama'],
                                   disabled=st.session_state.disabled)


    if not st.session_state['question_choices']:
        st.warning("Select a question")
    else:
        render_chat_interface()

def render_chat_interface():
    st.markdown("""
        <style>
            .message-container, .message-chatbot {
                padding: 10px;
                margin-top: 5px;
                border-radius: 5px;
            }
            .message-container {
                background-color: #f0f0f0;
                border-left: 5px solid #4CAF50;
            }
            .message-chatbot {
                background-color: #ffffff;
                border-left: 5px solid #002855;
            }
            .fixed-footer {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background-color: #fff;
                padding: 10px 200px;
                box-shadow: 0px -2px 10px rgba(0,0,0,0.1);
                z-index: 100;
            }
            .streamlit-container {
                padding-bottom: 70px;
            }
            .label {
                font-weight: bold;
                display: block;
                margin-bottom: 5px;
            }
            .message-text {
                margin-left: 20px;
            }
            .thread-container {
                margin-top: 20px;
                margin-bottom: 20px;
            }
            .stButton > button {
                width: 100%;
                border-radius: 10px;
                background-color: #002855;
                color: white;
            }
        </style>
        """, unsafe_allow_html=True)

    for message in st.session_state['message_history']:
        class_name = "message-container" if message['sender'] == '👤User' else "message-chatbot"
        label = message['sender']

        paragraphs = message["text"].split('\n')
        paragraph_html = ''.join(f'<p>{paragraph}</p>' for paragraph in paragraphs if paragraph.strip())

        st.markdown(f'''
        <div class="thread-container">
            <div class="{class_name}">
                <div class="label">{label}:</div>
                <div class="message-text">{paragraph_html}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="fixed-footer">', unsafe_allow_html=True)
    st.chat_input("Type your query here:", key="query", on_submit=ask_question)
    st.markdown('</div>', unsafe_allow_html=True)

    st.button("Start New Chat", key='start_new_chat', on_click = start_new_chat)

def ask_question():
    user_query = st.session_state.query
    if user_query:
        st.session_state['message_history'].append({'sender': '👤User', 'text': user_query})
        with st.spinner('Crafting response...'):
            response = check_fact(st.session_state['current_question'], user_query)
            st.session_state['message_history'].append({'sender': '🤖Chatbot', 'text': response})

def start_new_chat():
    st.session_state['message_history'] = []
    st.session_state['current_question'] = None
    st.session_state['question_choices'] = None
    st.session_state.disabled = False

if __name__ == '__main__':
    create_streamlit_interface()
