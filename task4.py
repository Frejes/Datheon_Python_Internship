"""Task 4: Interactive full-stack AI chatbot with Streamlit + LangChain.

Features: streaming token output, chat history, session-state memory,
swappable LLM provider, and a retrieval hook to plug in your RAG pipeline.

Run:  streamlit run task4.py
"""
import html
import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(page_title="Datheon AI", page_icon="💬", layout="wide")

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. If context is provided, use it to "
    "answer and say when the context does not contain the answer.\n\n"
    "Context:\n{context}"
)


# ---------- Plug your Task 1-3 RAG / agent pipeline in here ----------
def retrieve_context(query: str) -> str:
    """Return retrieved text for `query`. Replace with your retriever, e.g.:

        docs = vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(query)
        return "\n\n".join(d.page_content for d in docs)
    """
    return "No external context provided."


# ---------- Session state ----------
def init_state():
    if "messages" not in st.session_state:
        # Each item: {"role": "user" | "assistant", "content": str}
        st.session_state.messages = []
    if "appearance" not in st.session_state:
        st.session_state.appearance = "Light"


def clear_chat():
    st.session_state.messages = []


# ---------- LLM + chain ----------
def build_llm(provider: str, model: str, temperature: float, api_key: str):
    if provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, temperature=temperature, api_key=api_key, streaming=True
        )
    from langchain_openai import ChatOpenAI

    if provider == "OpenRouter":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            streaming=True,
        )
    return ChatOpenAI(
        model=model, temperature=temperature, api_key=api_key, streaming=True
    )


def build_chain(llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def to_lc_history(messages, max_turns: int):
    """Convert stored messages to LangChain messages, keeping the last N turns."""
    trimmed = messages[-max_turns * 2 :]
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in trimmed
    ]


# ---------- UI ----------
init_state()
theme_colors = {
    "Light": {
        "background": "#f7f8fc",
        "surface": "#ffffff",
        "surface_muted": "#f0f1f7",
        "text": "#202235",
        "border": "rgba(35, 38, 61, 0.14)",
    },
    "Dark": {
        "background": "#11131c",
        "surface": "#1b1e2a",
        "surface_muted": "#242737",
        "text": "#edeef8",
        "border": "rgba(230, 232, 255, 0.16)",
    },
}
theme = theme_colors[st.session_state.appearance]

theme_css = """
    <style>
    :root {
        --chat-accent: #6958e8;
        --chat-accent-soft: rgba(105, 88, 232, 0.12);
        --chat-background: __BACKGROUND__;
        --chat-surface: __SURFACE__;
        --chat-surface-muted: __SURFACE_MUTED__;
        --chat-text: __TEXT__;
        --chat-border: __BORDER__;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 50% -18%, var(--chat-accent-soft), transparent 52%),
            var(--chat-background);
        color: var(--chat-text);
    }

    .stApp label,
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stMarkdownContainer"] {
        color: var(--chat-text);
    }

    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background: var(--chat-background);
    }

    [data-testid="stSidebar"] {
        background: var(--chat-surface);
        color: var(--chat-text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid var(--chat-border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 7rem;
    }

    .chat-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .chat-brand-mark {
        display: grid;
        width: 2.8rem;
        height: 2.8rem;
        place-items: center;
        border: 1px solid var(--chat-border);
        border-radius: 0.95rem;
        background: var(--chat-accent-soft);
        color: var(--chat-accent);
        font-size: 1.35rem;
        font-weight: 700;
    }

    .chat-brand h1 {
        margin: 0;
        font-size: clamp(1.7rem, 3vw, 2.25rem);
        letter-spacing: -0.045em;
    }

    .chat-subtitle {
        margin: 0.25rem 0 1.6rem 3.55rem;
        color: var(--chat-text);
        opacity: 0.68;
        font-size: 0.95rem;
    }

    .chat-model {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        max-width: 100%;
        margin: 0 0 0 3.55rem;
        padding: 0.35rem 0.65rem;
        border: 1px solid var(--chat-border);
        border-radius: 999px;
        background: var(--chat-surface);
        color: var(--chat-text);
        font-size: 0.78rem;
        overflow-wrap: anywhere;
    }

    .chat-model-dot {
        color: #38a879;
    }

    .welcome-card {
        margin: 2rem 0 1.25rem;
        padding: 1.5rem;
        border: 1px solid var(--chat-border);
        border-radius: 1.15rem;
        background: var(--chat-surface);
    }

    .welcome-card h2 {
        margin: 0 0 0.4rem;
        font-size: 1.25rem;
        letter-spacing: -0.025em;
    }

    .welcome-card p {
        margin: 0;
        color: var(--chat-text);
        opacity: 0.68;
        line-height: 1.55;
    }

    .suggestion-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }

    .suggestion-card {
        min-height: 5.2rem;
        padding: 0.9rem 1rem;
        border: 1px solid var(--chat-border);
        border-radius: 0.9rem;
        background: var(--chat-surface-muted);
        color: var(--chat-text);
        font-size: 0.9rem;
        line-height: 1.45;
        opacity: 0.86;
    }

    .suggestion-card strong {
        display: block;
        margin-bottom: 0.3rem;
        color: var(--chat-accent);
        font-size: 0.76rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    [data-testid="stChatMessage"] {
        margin-bottom: 0.7rem;
        padding: 1rem 1.15rem;
        border: 1px solid var(--chat-border);
        border-radius: 1rem;
        background: var(--chat-surface);
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.65;
    }

    [data-testid="stChatInput"] {
        border-color: var(--chat-border);
        border-radius: 1rem;
        background: var(--chat-surface);
    }

    [data-testid="stChatInput"] > div {
        border-color: var(--chat-border) !important;
        background: var(--chat-surface) !important;
    }

    [data-testid="stChatInput"] textarea,
    [data-testid="stTextInput"] input {
        -webkit-text-fill-color: var(--chat-text);
        color: var(--chat-text);
        background: var(--chat-surface) !important;
    }

    [data-testid="stChatInput"] button {
        color: var(--chat-text);
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--chat-text);
        opacity: 0.58;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--chat-accent);
        box-shadow: 0 0 0 2px var(--chat-accent-soft);
    }

    [data-testid="stSidebar"] h2 {
        letter-spacing: -0.025em;
    }

    [data-testid="stSidebar"] [data-testid="stButton"] button {
        border-radius: 0.75rem;
    }

    @media (max-width: 640px) {
        .block-container {
            padding: 1.5rem 1rem 6rem;
        }

        .chat-subtitle {
            margin-left: 0;
        }

        .chat-model {
            margin-left: 0;
        }

        .suggestion-grid {
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }

        .suggestion-card {
            min-height: auto;
        }
    }
    </style>
"""
for token, color in (
    ("__BACKGROUND__", theme["background"]),
    ("__SURFACE__", theme["surface"]),
    ("__SURFACE_MUTED__", theme["surface_muted"]),
    ("__TEXT__", theme["text"]),
    ("__BORDER__", theme["border"]),
):
    theme_css = theme_css.replace(token, color)
st.markdown(theme_css, unsafe_allow_html=True)

with st.sidebar:
    st.title("Settings")
    st.caption("Tune how the assistant responds.")
    st.selectbox("Appearance", ["Light", "Dark"], key="appearance")
    provider = st.selectbox("Provider", ["OpenRouter", "Anthropic", "OpenAI"])
    defaults = {
        "OpenRouter": ("recursive/flint-v1.0:free", "OPENROUTER_API_KEY"),
        "Anthropic": ("claude-sonnet-4-6", "ANTHROPIC_API_KEY"),
        "OpenAI": ("gpt-4o-mini", "OPENAI_API_KEY"),
    }
    default_model, env_var = defaults[provider]
    # key= makes the widgets reset cleanly when the provider changes
    model = st.text_input("Model", value=default_model, key=f"model_{provider}")
    api_key = st.text_input(
        env_var,
        type="password",
        value=os.getenv(env_var, ""),
        key=f"key_{provider}",
    ).strip()
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_turns = st.slider("Memory (past turns sent to model)", 1, 20, 8)
    use_rag = st.toggle("Use retrieval (RAG)", value=False)
    if use_rag:
        st.caption("Retrieval is enabled; connect your retriever in `retrieve_context()`.")
    st.divider()
    st.button("🗑️ Clear conversation", on_click=clear_chat, use_container_width=True)

title_column, action_column = st.columns([5, 1.4], vertical_alignment="center")
with title_column:
    st.markdown(
        """
        <div class="chat-brand">
            <div class="chat-brand-mark">✦</div>
            <h1>Datheon AI</h1>
        </div>
        <p class="chat-subtitle">A thoughtful assistant for questions, ideas, and research.</p>
        """,
        unsafe_allow_html=True,
    )
    active_model = html.escape(model.strip() or default_model)
    st.markdown(
        f'<div class="chat-model"><span class="chat-model-dot">●</span>'
        f'{html.escape(provider)} · {active_model}</div>',
        unsafe_allow_html=True,
    )
with action_column:
    st.button("＋ New chat", on_click=clear_chat, use_container_width=True)

# Replay history on every rerun (Streamlit re-executes the script per interaction)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    st.markdown(
        """
        <section class="welcome-card">
            <h2>What would you like to work on?</h2>
            <p>Ask a question to get started. Your conversation stays in this session.</p>
        </section>
        <div class="suggestion-grid">
            <div class="suggestion-card"><strong>Explore</strong>Explain a complex idea in simple terms</div>
            <div class="suggestion-card"><strong>Create</strong>Help me draft or improve some writing</div>
            <div class="suggestion-card"><strong>Solve</strong>Break a tricky problem into clear steps</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if prompt := st.chat_input("Ask me anything..."):
    if not api_key:
        st.error(f"Please enter your {env_var} in the sidebar.")
        st.stop()

    history = to_lc_history(st.session_state.messages, max_turns)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            chain = build_chain(build_llm(provider, model, temperature, api_key))
            context = retrieve_context(prompt) if use_rag else "No external context provided."
            stream = chain.stream(
                {"context": context, "history": history, "question": prompt}
            )
            # write_stream renders tokens live and returns the full text
            response = st.write_stream(stream)
        except Exception as e:
            st.error(f"⚠️ {e}")
            # Remove the failed question so errors never pollute the memory
            st.session_state.messages.pop()
            st.stop()

    st.session_state.messages.append({"role": "assistant", "content": response})