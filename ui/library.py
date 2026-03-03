from time import time
import streamlit as st

@st.dialog('Confirm library reset')
def reset_dialog(vector_db):
    st.warning('This will delete all images from the library. Are you sure you want to proceed?')
    col1, col2 = st.columns(2, width=300)
    if col1.button('Proceed', type='primary', width=150):
        vector_db.flush()
        st.rerun(scope='fragment')
    if col2.button('Cancel', type='secondary', width=150):
        st.rerun(scope='fragment')

@st.fragment
def library_tab(engine, vector_db):
    st.header('Add images to the library', divider=True)
    # defining basic layout
    col1, col2, col3 = st.columns([0.70, 0.15, 0.15])
    dir_path_input = col1.text_input('Enter the directory path:', placeholder='/dir/images',
                                     label_visibility='collapsed')

    if col2.button('Scan directory', type='primary', disabled=not dir_path_input):
        progress_bar = col1.progress(0, text='Scanning directory...')
        start = time()
        num_images = engine.extract_image_embeddings(
            dir_path_input,
            vector_db,
            progress_bar,
            st.session_state['settings']['batch_size'],
            st.session_state['settings']['num_workers']
        )
        stop = time()
        progress_bar.progress(1.0, text='Done')
        if num_images == 0:
            response_text = 'No new images found in the directory'
        else:
            response_text = f'Added {num_images} new images to the library in {stop - start:.2f} seconds ({num_images / (stop - start):.2f} images/sec)'
        col1.success(response_text, icon='✅')
    if col3.button('Reset library', type='primary'):
        reset_dialog(vector_db=vector_db)

    st.header('Library statistics', divider=True)
    col1, col2 = st.columns(2, width=500)
    col1.metric('Total images', vector_db.total_images)
    col2.metric('Database memory usage', vector_db.total_memory_usage)
