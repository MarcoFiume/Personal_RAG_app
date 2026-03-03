import streamlit as st
from utils import save_settings

def sidebar():
    with st.sidebar:
        st.header('⚙️ Settings')

        st.subheader('Selected model:', divider=True)
        st.write(st.session_state["settings"]["model_id"])

        with st.expander('Image embeddings extraction:'):
            st.session_state['settings']['batch_size'] = st.slider(
                "Batch size", 1, 32, st.session_state['settings']['batch_size'],
                help="Number of images to be loaded on the gpu for at once. (increase cpu and gpu memory usage)"
            )
            st.session_state['settings']['num_workers'] = st.slider(
                "Num workers", 1, 8, st.session_state['settings']['num_workers'],
                help="Number of parallel workers to use for image loading. (increase cpu memory usage)"
            )

        with st.expander('Image search:'):
            st.session_state['settings']['top_k'] = st.number_input(
                'Default Top K',
                1,
                st.session_state['settings']['max_top_k'],
                st.session_state['settings']['top_k'],
                help='Number of results to return for each search query.'
            )
            st.session_state['settings']['page_size'] = st.number_input(
                'Default Page Size',
                1,
                st.session_state['settings']['max_page_size'],
                st.session_state['settings']['page_size'],
                help='Number of results to return per page.'
            )
            st.info('The maximum values for top_k and page_size can be changed in the settings.json file.')

        st.button("Save settings", on_click=save_settings(st.session_state['settings']), type="primary")
