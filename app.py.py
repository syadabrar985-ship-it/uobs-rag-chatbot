import os
import streamlit as st

# Make the Streamlit secret available to the existing Stage 5 backend.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

from stage5_rag import ask_uobs, initialize_rag


st.set_page_config(
    page_title="UoBS AI Assistant",
    page_icon="🎓",
)

st.title("UoBS AI Assistant")
st.caption(
    "Your AI assistant for information about the University of Baltistan, Skardu."
)


@st.cache_resource
def load_rag():
    return initialize_rag()


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("UoBS AI Assistant")
    st.write(
        "This chatbot uses Retrieval-Augmented Generation (RAG) "
        "to answer questions using the available University of Baltistan data."
    )

    st.subheader("Technology")
    st.write(
        "- Python\n"
        "- Streamlit\n"
        "- Sentence Transformers\n"
        "- FAISS\n"
        "- Groq LLM"
    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if not st.session_state.messages:
    st.info(
        "Welcome to the UoBS AI Assistant. "
        "Ask me anything about the University of Baltistan, Skardu."
    )

    st.subheader("Example questions")
    st.write("• What programs does the University of Baltistan offer?")
    st.write("• What are the admission requirements?")
    st.write("• What faculties are available?")
    st.write("• Where is the University of Baltistan located?")
    st.write("• How can I contact the university?")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            st.markdown("**Sources**")
            for source in message["sources"]:
                st.markdown(f"- {source}")


user_input = st.chat_input("Ask a question about UoBS...")

if user_input and user_input.strip():
    question = user_input.strip()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
            "sources": [],
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the UoBS knowledge base..."):
            try:
                index, metadata, embedding_model, groq_client = load_rag()

                result = ask_uobs(
                    question,
                    index,
                    metadata,
                    embedding_model,
                    groq_client,
                )

                answer = str(result.get("answer", "")).strip()
                sources = result.get("sources", []) or []

                if not answer:
                    answer = (
                        "I couldn't find a reliable answer in the available "
                        "University of Baltistan data."
                    )

                st.markdown(answer)

                if sources:
                    st.markdown("**Sources**")
                    for source in sources:
                        st.markdown(f"- {source}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception:
                st.error(
                    "Sorry, I couldn't process your question right now. "
                    "Please try again."
                )