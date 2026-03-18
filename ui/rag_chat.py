import time
import streamlit as st


def stream_messages(response):
    for char in response:
        yield char
        time.sleep(0.05)


def rag_chat_tab(engine, vector_db):
    st.header("Ask questions about your documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    messages_container = st.container(height=600, border=True)

    with messages_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        if prompt == 'clear':
            st.session_state.messages.clear()
        else:
            with messages_container.chat_message("user"):
                st.markdown(prompt)
            response = prompt
            time.sleep(0.6)
            with messages_container.chat_message("assistant"):
                response = st.write_stream(stream_messages(response))
            st.session_state.messages.append({"role": "assistant", "content": response})
