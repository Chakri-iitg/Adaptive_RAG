import streamlit as st
import requests

url = "http://localhost:8000/api/chat"

st.set_page_config(page_title="Adaptive RAG", layout="wide")
st.title("Adaptive RAG Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    role = m["role"]
    content = m["content"]
    with st.chat_message(role):
        st.markdown(content)

if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        resp = requests.post(url, json={"question":prompt})

        if resp.status_code == 200:
            data = resp.json()
            answer = data.get("answer", "No answer")
            docs = data.get("documents",[])
            st.session_state.messages.append({"role":"assistant","content":answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
            with st.expander("Retrieved documents"):
                for i, d in enumerate(docs, 1):
                    st.markdown(f"**Doc {i}:** {d[:500]}")
        else:
            st.error(f"API Error: {resp.status_code}")
    except Exception as e:
        st.error(f"Backend error: {e}")