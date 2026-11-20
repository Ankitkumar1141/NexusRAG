import logging
import streamlit as st
import config
from src.cache import clear as clear_query_cache, stats as cache_stats
from src.indexing import (
    build_index,
    clear_index,
    index_exists,
    index_summary,
    update_index,
)
from src.ingestion import SUPPORTED_EXTENSIONS, load_document_from_folder
from src.memory import add_exchange, clear_memory, create_memory
from src.rag_pipeline import answer_question, load_pipeline
from src.reranking import reranker_status


logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("src").setLevel(logging.INFO)

st.set_page_config(page_title="NexusRAG", page_icon="📚", layout="wide")


def init_session_state() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = create_memory()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


def get_pipeline():
    if st.session_state.pipeline is None:
        try:
            with st.spinner("Loading the index..."):
                st.session_state.pipeline = load_pipeline()
        except Exception as error:
            st.error(f"Could not load the index: {error}")
            return None

    return st.session_state.pipeline


def save_uploaded_files(uploaded_files) -> list:
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for uploaded_file in uploaded_files:
        destination = config.DOCUMENTS_DIR / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(destination)

    return saved_paths


def render_sidebar() -> bool:
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Changing this only affects the running app. To make it permanent
        # (and to use the command-line scripts) edit VECTOR_DB in your .env.
        options = ["faiss", "pinecone"]
        chosen_db = st.selectbox(
            "Vector database",
            options,
            index=options.index(config.VECTOR_DB) if config.VECTOR_DB in options else 0,
            help="FAISS runs locally. Pinecone needs an API key in your .env file.",
        )
        if chosen_db != config.VECTOR_DB:
            config.VECTOR_DB = chosen_db
            st.session_state.pipeline = None

        for name, value in config.check_model().items():
            st.caption(f"**{name}:** {value}")

        st.caption(f"**Reranker:** {reranker_status()}")

        # Cache stats
        cs = cache_stats()
        st.caption(
            f"**Query Cache:** {cs['size']} entries · "
            f"{cs['hits']} hits · {cs['misses']} misses"
        )

        st.divider()
        render_document_management()

        st.divider()
        st.header("💬 Chat controls")

        if st.button("Clear conversation", width="stretch"):
            clear_memory(st.session_state.memory)
            st.session_state.messages = []
            st.rerun()

        if st.button("🗑️ Clear query cache", width="stretch", help="Wipe the in-memory answer cache"):
            clear_query_cache()
            st.success("Query cache cleared.")

        show_debug = st.toggle(
            "Show Retrieval Details",
            value=False,
            help="See the queries, retrieved chunks, reranked chunks and final context.",
        )

    return show_debug


def render_document_management() -> None:
    st.header("📄 Documents")

    if index_exists():
        summary = index_summary()
        st.success(
            f"Index ready: {summary['children']} child chunks "
            f"from {len(summary['files'])} file(s)."
        )
        with st.expander("Indexed files"):
            for filename in summary["files"]:
                st.write(f"- {filename}")
    else:
        st.info("No index yet. Build one to start asking questions.")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Save uploaded files", width="stretch"):
        saved = save_uploaded_files(uploaded_files)
        st.success(f"Saved {len(saved)} file(s) to data/documents.")

    if st.button("🔨 Build Vector DB", width="stretch"):
        run_build()

    if st.button("➕ Update Vector DB", width="stretch"):
        run_update()

    if st.button("🗑️ Clear / Rebuild", width="stretch"):
        clear_index()
        clear_query_cache()          # Stale answers must not be served after a rebuild
        st.session_state.pipeline = None
        st.success("Index cleared. Click 'Build Vector DB' to start again.")


def run_build() -> None:
    try:
        with st.spinner("Loading documents, chunking and embedding..."):
            documents = load_document_from_folder(config.DOCUMENTS_DIR)
            stats = build_index(documents)

        clear_query_cache()          # New index → old cached answers are stale
        st.session_state.pipeline = None
        st.success(
            f"Indexed {stats['documents']} pages -> {stats['parents']} parent chunks "
            f"-> {stats['children']} child chunks."
        )
    except Exception as error:
        st.error(f"Build failed: {error}")


def run_update() -> None:
    try:
        with st.spinner("Adding new documents to the index..."):
            documents = load_document_from_folder(config.DOCUMENTS_DIR)
            stats = update_index(documents)

        clear_query_cache()          # New documents may change answers to known questions
        st.session_state.pipeline = None
        if "added_children" in stats:
            st.success(
                f"Added {stats['added_children']} new chunks "
                f"(skipped {stats['skipped_children']} already indexed)."
            )
        else:
            st.success(f"Built a new index with {stats['children']} chunks.")
    except Exception as error:
        st.error(f"Update failed: {error}")


def render_debug_panel(debug: dict) -> None:
    with st.expander("🔍 Retrieval details", expanded=False):

        if debug.get("cache_hit"):
            st.success("⚡ Served from query cache — no retrieval or LLM call needed.")
            return

        st.markdown("**1. Original query**")
        st.write(debug["original_query"])

        st.markdown("**2. Generated queries (multi-query retrieval)**")
        for query in debug["generated_queries"]:
            st.write(f"- {query}")

        st.markdown(
            f"**3. Retrieved candidates after hybrid search "
            f"({len(debug['retrieved'])} chunks)**"
        )
        show_chunks(debug["retrieved"])

        st.markdown(
            f"**4. After cross-encoder reranking ({len(debug['reranked'])} chunks)**"
        )
        show_chunks(debug["reranked"])

        st.markdown(
            f"**5. Final context sent to the LLM ({len(debug['final_context'])} chunks)**"
        )
        show_chunks(debug["final_context"])


def show_chunks(chunks: list, limit: int = 5) -> None:
    if not chunks:
        st.caption("_(none)_")
        return

    for number, chunk in enumerate(chunks[:limit], start=1):
        score = f" · score {chunk['score']}" if chunk["score"] is not None else ""
        st.caption(f"{number}. **{chunk['source']}**{score}")
        st.text(chunk["preview"][:300])

    if len(chunks) > limit:
        st.caption(f"_...and {len(chunks) - limit} more_")


def render_sources(sources: list) -> None:
    if not sources:
        return

    lines = [f"{number}. {source}" for number, source in enumerate(sources, start=1)]
    st.markdown("**Sources:**\n\n" + "\n".join(lines))


def main() -> None:
    init_session_state()

    st.title("📚 NexusRAG")

    show_debug = render_sidebar()

    if not config.MISTRAL_API_KEY:
        st.warning(
            "No MistralAI API key found. Copy `.env.example` to `.env`, add your "
            "key, then restart the app."
        )
        return

    if not index_exists():
        st.info(
            "Nothing is indexed yet. Use **Build Vector DB** in the sidebar to "
            "index the sample TechNova documents (or upload your own first)."
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))
                if show_debug and message.get("debug"):
                    render_debug_panel(message["debug"])

    question = st.chat_input("Ask a question about your documents...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    pipeline = get_pipeline()
    if pipeline is None:
        return

    with st.chat_message("assistant"):
        cache_hit = False
        with st.spinner("Searching documents and writing an answer..."):
            try:
                result = answer_question(question, pipeline, st.session_state.memory)
                cache_hit = result.get("debug", {}).get("cache_hit", False)
            except Exception as error:
                st.error(f"Something went wrong: {error}")
                return

        if cache_hit:
            st.caption("⚡ _Answered from cache_")

        st.markdown(result["answer"])
        render_sources(result["sources"])
        if show_debug:
            render_debug_panel(result["debug"])

    add_exchange(st.session_state.memory, question, result["answer"])
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "debug": result["debug"],
        }
    )


if __name__ == "__main__":
    main()
