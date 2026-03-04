import streamlit as st
from utils import save_settings

@st.fragment
def sidebar():
    st.header('⚙️ Settings')

    st.subheader('Selected model:', divider=True)
    st.write(st.session_state['settings']['model_id'])

    with st.expander('Image embeddings extraction:'):
        st.session_state['settings']['batch_size'] = st.slider(
            'Batch size', 1, 32, st.session_state['settings']['batch_size'],
            help='Number of images to be loaded on the gpu for at once. (increase cpu and gpu memory usage)'
        )
        st.session_state['settings']['num_workers'] = st.slider(
            'Num workers', 1, 8, st.session_state['settings']['num_workers'],
            help='Number of parallel workers to use for image loading. (increase cpu memory usage)'
        )

    with st.expander('Database backend:'):
        show_warning = False
        st.session_state['settings']['vector_db_backend'] = st.selectbox(
            'Backend', ['sqlite', 'redis'],
            index=0 if st.session_state['settings']['vector_db_backend'] == 'sqlite' else 1,
            help='Choose the vector database backend.'
        )
        if st.session_state['settings']['vector_db_backend'] == 'sqlite':
            st.session_state['settings']['sqlite_db_path'] = st.text_input(
                'SQLite DB path', st.session_state['settings']['sqlite_db_path']
            )
        else:
            st.session_state['settings']['redis_host'] = st.text_input(
                'Redis host', st.session_state['settings']['redis_host']
            )
            st.session_state['settings']['redis_port'] = st.number_input(
                'Redis port', value=st.session_state['settings']['redis_port']
            )
        st.warning('Is necessary to save the settings for changes to take effect.')

    with st.expander('Image search:'):
        st.session_state['settings']['max_top_k'] = st.number_input(
            'Max top k',
            1,
            value=st.session_state['settings']['max_top_k'],
            help='Maximum number of results to return for each search query.'
        )
        st.session_state['settings']['max_page_size'] = st.number_input(
            'Max page size',
            1,
            value=st.session_state['settings']['max_page_size'],
            help='Maximum number of results to return per page.'
        )


    if st.button('Save settings', type='primary'):
        save_settings(st.session_state['settings'])
        st.rerun(scope='app')
