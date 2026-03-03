import streamlit as st
from inference import ImgEmbeddingEngine
from db import VectorDatabase
from ui import sidebar, library_tab, search_tab
from utils import load_settings

@st.cache_resource(show_spinner='Loading model...')
def load_engine(model_id):
    return ImgEmbeddingEngine(model_id)

@st.cache_resource(show_spinner='Loading vector database...')
def load_vector_db(index_name, vector_dim):
    return VectorDatabase(index_name, vector_dim)

st.session_state['settings'] = load_settings()
st.set_page_config(page_title='RAG app', layout='wide', menu_items={'about' : 'About this app'})
st.title("Retrieve augmented generation demo app")
sidebar()
st.divider()

engine = load_engine(st.session_state['settings']['model_id'])
vector_db = load_vector_db(
    st.session_state['settings']['index_name'],
    engine.vector_dim
)

tab1, tab2 = st.tabs(['Manage image library', 'Search images'])

with tab1:
    library_tab(engine, vector_db)

with tab2:
    search_tab(engine, vector_db)
