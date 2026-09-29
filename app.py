import os
import hashlib
import tempfile

import streamlit as st
import faiss
import nltk

from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter

from sentence_transformers import SentenceTransformer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 PDF RAG Chatbot")

st.caption(
    "Upload a PDF and ask questions using Retrieval-Augmented Generation"
)


# ============================================================
# NLTK SETUP
# ============================================================

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "rag_data" not in st.session_state:
    st.session_state.rag_data = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    st.divider()

    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    st.divider()

    if st.button(
        "🧹 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    try:

        api_key = st.secrets["GEMINI_API_KEY"]

    except Exception:

        st.error(
            "GEMINI_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

        st.stop()

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# BUILD RAG INDEX
# ============================================================

@st.cache_resource(show_spinner=False)
def build_rag(pdf_bytes):

    # --------------------------------------------------------
    # Save PDF temporarily
    # --------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(pdf_bytes)

        temp_pdf_path = temp_file.name


    try:

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(
            temp_pdf_path
        )

        documents = loader.load()


        # ----------------------------------------------------
        # Sentence-based chunking
        # ----------------------------------------------------

        text_splitter = NLTKTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_documents(
            documents
        )


        # ----------------------------------------------------
        # Extract chunk text
        # ----------------------------------------------------

        chunk_texts = [
            chunk.page_content
            for chunk in chunks
        ]


        # ----------------------------------------------------
        # Embedding model
        # ----------------------------------------------------

        embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        embeddings = embedding_model.encode(
            chunk_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )


        # ----------------------------------------------------
        # Normalize embeddings
        #
        # Inner Product on normalized vectors
        # ≈ Cosine Similarity
        # ----------------------------------------------------

        faiss.normalize_L2(
            embeddings
        )


        # ----------------------------------------------------
        # Create FAISS index
        # ----------------------------------------------------

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(
            dimension
        )

        index.add(
            embeddings
        )


        # ----------------------------------------------------
        # Return RAG components
        # ----------------------------------------------------

        return {
            "chunks": chunks,
            "embedding_model": embedding_model,
            "index": index,
            "dimension": dimension
        }

    finally:

        # ----------------------------------------------------
        # Remove temporary PDF
        # ----------------------------------------------------

        if os.path.exists(temp_pdf_path):

            os.remove(
                temp_pdf_path
            )


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    pdf_name = uploaded_file.name

    # Create a unique hash based on actual PDF content
    pdf_hash = hashlib.md5(
        pdf_bytes
    ).hexdigest()


    # Rebuild RAG only when the actual PDF changes
    if (
        st.session_state.pdf_hash != pdf_hash
        or st.session_state.rag_data is None
    ):

        with st.spinner(
            "Processing PDF: loading → chunking → embeddings → FAISS..."
        ):

            st.session_state.rag_data = build_rag(
                pdf_bytes
            )


        # Reset chat for a new document
        st.session_state.messages = []

        st.session_state.pdf_name = pdf_name

        st.session_state.pdf_hash = pdf_hash


        st.success(
            f"PDF processed successfully: {pdf_name}"
        )


# ============================================================
# STOP IF NO PDF
# ============================================================

if st.session_state.rag_data is None:

    st.info(
        "👈 Upload a PDF from the sidebar to start chatting."
    )

    st.stop()


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

rag = st.session_state.rag_data

chunks = rag["chunks"]

embedding_model = rag["embedding_model"]

index = rag["index"]


# ============================================================
# SIDEBAR DOCUMENT INFORMATION
# ============================================================

with st.sidebar:

    st.success(
        f"📄 {st.session_state.pdf_name}"
    )

    st.write(
        f"**Total chunks:** {len(chunks)}"
    )

    st.write(
        f"**Embedding dimension:** {rag['dimension']}"
    )

    st.write(
        f"**FAISS vectors:** {index.ntotal}"
    )

    st.write(
        "**LLM:** Gemini 3.8 Flash"
    )

    st.write(
        "**Embeddings:** all-MiniLM-L6-v2"
    )


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def search_pdf(
    query,
    top_k=3
):

    # --------------------------------------------------------
    # Convert user query into embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )


    # --------------------------------------------------------
    # Normalize query embedding
    # --------------------------------------------------------

    faiss.normalize_L2(
        query_embedding
    )


    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )


    results = []


    # --------------------------------------------------------
    # Prepare retrieval results
    # --------------------------------------------------------

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        # Safety check
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]


        results.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": int(idx),
                "page": chunk.metadata.get("page"),
                "source": chunk.metadata.get("source"),
                "text": chunk.page_content
            }
        )


    return results


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful PDF question-answering assistant.

Your job is to answer questions using the retrieved context
from the uploaded PDF.

IMPORTANT RULES:

1. The uploaded PDF is the primary and authoritative source.

2. Use only information supported by the retrieved PDF context.

3. Do not invent facts.

4. Do not use outside knowledge to fill missing information.

5. Do not make unsupported assumptions or deductions.

6. If the requested information is not available in the
   retrieved PDF context, say exactly:

   "The requested information is not available in the uploaded PDF."

7. If the question asks for an explanation, explain only what
   can reasonably be supported by the retrieved context.

8. Preserve equations, numbers, terminology, names and facts
   from the PDF accurately.

9. Do not change the meaning of information from the PDF.

10. When the retrieved context contains a page number, mention
    the page number when useful.

11. Previous conversation messages may help understand the
    current question, but document facts must come from the
    retrieved PDF context.

12. If the PDF context is insufficient, do not guess.

13. Keep the answer clear, concise and relevant.
"""


# ============================================================
# CREATE RETRIEVED CONTEXT
# ============================================================

def create_context(results):

    context_parts = []


    for result in results:

        page = result["page"]


        if page is not None:

            page_number = page + 1

        else:

            page_number = "Unknown"


        context_parts.append(
            f"""
[Retrieved Chunk {result["rank"]}]
[Chunk ID: {result["chunk_id"]}]
[Page: {page_number}]
[Similarity Score: {result["score"]:.4f}]

{result["text"]}
"""
        )


    return "\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER USING GEMINI
# ============================================================

def generate_answer(
    query,
    top_k=3
):

    # --------------------------------------------------------
    # Step 1: Retrieve relevant chunks
    # --------------------------------------------------------

    results = search_pdf(
        query,
        top_k
    )


    # --------------------------------------------------------
    # Step 2: Create context
    # --------------------------------------------------------

    context = create_context(
        results
    )


    # --------------------------------------------------------
    # Step 3: Add recent conversation
    # --------------------------------------------------------

    conversation = ""


    for message in st.session_state.messages[-6:]:

        conversation += (
            f"\n{message['role'].upper()}: "
            f"{message['content']}\n"
        )


    # --------------------------------------------------------
    # Step 4: Create augmented prompt
    # --------------------------------------------------------

    prompt = f"""
Retrieved PDF Context
=====================

{context}


Previous Conversation
=====================

{conversation}


Current User Question
=====================

{query}


Task
====

Answer the current user question using the retrieved
PDF context.

Do not use outside knowledge.

If the answer cannot be supported by the retrieved
PDF context, say:

"The requested information is not available in the uploaded PDF."
"""


    # --------------------------------------------------------
    # Step 5: Gemini client
    # --------------------------------------------------------

    client = get_gemini_client()


    # --------------------------------------------------------
    # Step 6: Generate answer
    # --------------------------------------------------------

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=f"""
{SYSTEM_PROMPT}

{prompt}
"""
    )


    # --------------------------------------------------------
    # Step 7: Extract answer
    # --------------------------------------------------------

    answer = response.text


    return answer, results


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask a question about your PDF..."
)


# ============================================================
# HANDLE USER QUESTION
# ============================================================

if user_question:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_question
        )


    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching PDF and generating answer..."
        ):

            try:

                answer, results = generate_answer(
                    user_question,
                    top_k
                )


                st.markdown(
                    answer
                )


            except Exception as e:

                answer = (
                    "Sorry, something went wrong.\n\n"
                    f"Error: {e}"
                )

                st.error(
                    answer
                )

                results = []


    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # ========================================================
    # SHOW RETRIEVED SOURCES
    # ========================================================

    if results:

        with st.expander(
            "🔎 View Retrieved Sources"
        ):

            for result in results:

                page = result["page"]


                if page is not None:

                    page = page + 1


                st.markdown(
                    f"""
### Chunk {result["rank"]}

**Chunk ID:** `{result["chunk_id"]}`

**Page:** `{page}`

**Similarity Score:**
`{result["score"]:.4f}`
"""
                )


                st.write(
                    result["text"]
                )


                st.divider()