import streamlit as st

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
            'Number of columns',
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
