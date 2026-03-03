from time import time
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

def img_viewer(img_paths, n_col=4):
    with st.container(height=1000, border=False):
        st.space('medium')
        for i in range(0, len(img_paths), n_col):
            row_batch = img_paths[i: i + n_col]
            cols = st.columns(n_col)
            for img, col in zip(row_batch, cols):
                col.image(img['path'], caption=f'{img["path"]}')

def pagination(container, total_results, page_size):
    current_page = st.session_state.last_search['page']
    first_col, prev_col, page_col, next_col, last_col = container.columns([0.1, 0.1, 0.4, 0.1, 0.1])
    page_col.button(f'Page {current_page + 1}', disabled=True, type='tertiary', width='stretch')
    new_page = None
    if first_col.button('First', disabled=current_page == 0, width='stretch'):
        new_page = 0
    if prev_col.button('← Prev', disabled=current_page == 0, width='stretch'):
        new_page = current_page - 1
    if next_col.button('Next →', disabled=current_page >= (total_results - 1) // page_size, width='stretch'):
        new_page = current_page + 1
    if last_col.button('Last', disabled=current_page >= (total_results - 1) // page_size, width='stretch'):
        new_page = (total_results - 1) // page_size
    return new_page

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

@st.fragment
def search_tab(engine, vector_db):
    st.header('Search for images based on text prompt')
    # defining basic layout
    col1, col2 = st.columns([0.75, 0.25])
    prompt = col1.text_input(
        'Enter a prompt:',
        placeholder='e.g. a red sports car on a mountain road',
        label_visibility="collapsed",
        key='prompt'
    )
    if 'last_search' not in st.session_state:
        st.session_state['last_search'] = {'prompt': '', 'top_k': 0, 'results': [], 'page': 0}

    with col2.popover('Advanced options'):
        st.subheader('Search options:')
        top_k = st.number_input(
            'Number of results',
            1,
            st.session_state['settings']['max_top_k'],
            st.session_state['settings']['top_k']
        )
        page_size = st.number_input(
            'Page size',
            1,
            st.session_state['settings']['max_page_size'],
            st.session_state['settings']['page_size'],
        )
        st.subheader('Image viewer layout:')
        n_col = st.slider(
            "Number of columns",
            2,
            st.session_state['settings']['img_viewer_max_cols'],
            st.session_state['settings']['img_viewer_cols'],
        )

    if prompt and (
            prompt != st.session_state.last_search['prompt']
            or top_k != st.session_state.last_search['top_k']
            ):
        text_embedding = engine.extract_text_embedding(prompt)
        results = vector_db.search(
            text_embedding, top_k=top_k, page=0, page_size=page_size
        )
        st.session_state.last_search = {'prompt': prompt, 'top_k': top_k, 'results': results, 'page': 0}

    if st.session_state.last_search['results']:
        new_page = pagination(col1, top_k, page_size)

        if col2.button('Clear search', type='primary', on_click=lambda: st.session_state.update({'prompt': ''})):
            st.session_state.last_search = {'prompt': '', 'top_k': top_k, 'results': [], 'page': 0}
            st.rerun(scope='fragment')

        img_viewer(st.session_state.last_search['results'], n_col=n_col)

        if new_page is not None:
            text_embedding = engine.extract_text_embedding(st.session_state.last_search['prompt'])
            results = vector_db.search(
                text_embedding, top_k=top_k, page=new_page, page_size=page_size
            )
            st.session_state.last_search.update({'results': results, 'page': new_page})
            st.rerun(scope='fragment')

    else:
        col1.info('No results found')