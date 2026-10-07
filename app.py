import os
import re
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Company Policy Assistant",
    page_icon="📋",
    layout="wide"
)

CSV_FILE = "company_policies.csv"

# Gemini models
GENERATION_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite"
)

EMBEDDING_MODEL = os.environ.get(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-001"
)

# Files used to store the local vector index
INDEX_FILE = "policy_embeddings.npy"
INDEX_METADATA_FILE = "policy_embeddings_metadata.json"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        margin-bottom: 1.5rem;
    }

    .method-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .metric-box {
        padding: 10px;
        border-radius: 8px;
        background-color: #f5f5f5;
        margin-bottom: 10px;
    }

    .policy-box {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 12px;
        background-color: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API INITIALIZATION
# ============================================================

def get_api_key():
    """
    Look for the Gemini API key in Streamlit secrets first,
    then environment variables.
    """
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    return os.environ.get("GEMINI_API_KEY")


def initialize_gemini():
    api_key = get_api_key()

    if not api_key:
        return None, (
            "Gemini API key not found. Add GEMINI_API_KEY to "
            "Streamlit secrets (.streamlit/secrets.toml) or as an environment variable."
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GENERATION_MODEL)
        return model, None

    except Exception as e:
        return None, f"Could not initialize Gemini: {str(e)}"


# ============================================================
# LOAD POLICY DATA
# ============================================================

@st.cache_data
def load_policies():
    """
    Load and validate the company policy CSV.
    """
    if not Path(CSV_FILE).exists():
        raise FileNotFoundError(
            f"Could not find {CSV_FILE}. "
            "Place it in the same folder as app.py."
        )

    df = pd.read_csv(CSV_FILE)

    required_columns = [
        "title",
        "department",
        "policy_text",
        "category"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"CSV is missing the following columns: {missing}"
        )

    for col in required_columns:
        df[col] = df[col].fillna("").astype(str)

    return df


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return set(normalize_text(text).split())


# ============================================================
# 1. RULES-BASED SEARCH
# ============================================================

def rules_based_search(question, df):
    start_time = time.perf_counter()
    question_tokens = tokenize(question)

    stopwords = {
        "the", "a", "an", "is", "are", "what", "how",
        "can", "i", "do", "does", "of", "to", "for",
        "and", "or", "in", "on", "my", "our", "company",
        "policy", "about", "please", "tell", "me"
    }

    question_tokens = {
        word for word in question_tokens
        if word not in stopwords and len(word) > 2
    }

    scores = []

    for _, row in df.iterrows():
        searchable_text = " ".join([
            row["title"],
            row["department"],
            row["category"],
            row["policy_text"]
        ])

        policy_tokens = tokenize(searchable_text)
        overlap = question_tokens.intersection(policy_tokens)
        score = len(overlap)

        title_tokens = tokenize(row["title"])
        title_overlap = question_tokens.intersection(title_tokens)
        score += len(title_overlap) * 3

        scores.append(score)

    df_temp = df.copy()
    df_temp["rule_score"] = scores

    best_idx = df_temp["rule_score"].idxmax()
    best_score = df_temp.loc[best_idx, "rule_score"]
    elapsed = time.perf_counter() - start_time

    if best_score == 0:
        return {
            "answer": "No clear policy match was found using keyword rules.",
            "policy": None,
            "time": elapsed,
            "tokens": 0,
            "support": "Not supported / no match",
            "score": 0
        }

    policy = df_temp.loc[best_idx]
    answer = f"According to the **{policy['title']}**, {policy['policy_text']}"
    estimated_tokens = max(1, len(answer.split()) * 4 // 3)

    return {
        "answer": answer,
        "policy": policy,
        "time": elapsed,
        "tokens": estimated_tokens,
        "support": "Supported by database",
        "score": best_score
    }


# ============================================================
# POLICY TEXT FOR LLM WITHOUT RAG
# ============================================================

def build_raw_policy_context(df, max_chars=80000):
    sections = []
    for i, row in df.iterrows():
        section = f"""
POLICY {i + 1}
Title: {row['title']}
Department: {row['department']}
Category: {row['category']}
Policy text: {row['policy_text']}
"""
        sections.append(section)

    context = "\n".join(sections)
    if len(context) > max_chars:
        context = context[:max_chars]
        context += "\n[Policy database truncated due to prompt size.]"
    return context


# ============================================================
# LLM RESPONSE PARSER
# ============================================================

def parse_llm_response(text):
    result = {
        "answer": text,
        "policy_title": "Not identified",
        "support": "Unclear"
    }

    policy_match = re.search(r"POLICY\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if policy_match:
        result["policy_title"] = policy_match.group(1).strip()

    support_match = re.search(r"SUPPORT\s*:\s*(.+)", text, flags=re.IGNORECASE)
    if support_match:
        result["support"] = support_match.group(1).strip()

    cleaned = re.sub(r"^\s*POLICY\s*:.*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"^\s*SUPPORT\s*:.*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    result["answer"] = cleaned.strip()

    return result


# ============================================================
# 2. LLM WITHOUT VECTOR INDEX
# ============================================================

def llm_without_rag(question, df, model):
    start_time = time.perf_counter()
    context = build_raw_policy_context(df)

    prompt = f"""
You are a company policy assistant.
You have access to the company's policy database below.
Answer the employee's question using ONLY the information contained in the policy database.
Do not invent policies or rules.
If the database does not contain enough information, say so clearly.
You must identify the policy that is most relevant to the question.

At the END of your response, include exactly these two lines:
POLICY: [most relevant policy title]
SUPPORT: [Supported / Not supported / Unclear]

EMPLOYEE QUESTION:
{question}

COMPANY POLICY DATABASE:
{context}
"""

    try:
        response = model.generate_content(prompt)
        text = response.text
        parsed = parse_llm_response(text)
        elapsed = time.perf_counter() - start_time
        estimated_tokens = max(1, len(text.split()) * 4 // 3)

        return {
            "answer": parsed["answer"],
            "policy_title": parsed["policy_title"],
            "policy": find_policy_by_title(parsed["policy_title"], df),
            "time": elapsed,
            "tokens": estimated_tokens,
            "support": parsed["support"]
        }
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return {
            "answer": f"Gemini API error: {str(e)}",
            "policy_title": "Error",
            "policy": None,
            "time": elapsed,
            "tokens": 0,
            "support": "Unavailable"
        }


def find_policy_by_title(title, df):
    if not title:
        return None
    title_normalized = normalize_text(title)
    for _, row in df.iterrows():
        if normalize_text(row["title"]) == title_normalized:
            return row
    for _, row in df.iterrows():
        if title_normalized in normalize_text(row["title"]) or normalize_text(row["title"]) in title_normalized:
            return row
    return None


# ============================================================
# VECTOR INDEX & RAG
# ============================================================

def create_embeddings(df):
    embeddings = []
    for _, row in df.iterrows():
        text = f"Title: {row['title']}\nDepartment: {row['department']}\nCategory: {row['category']}\nPolicy: {row['policy_text']}"
        result = genai.embed_content(model=EMBEDDING_MODEL, content=text)
        embeddings.append(result["embedding"])
    return np.array(embeddings, dtype=np.float32)


def save_vector_index(embeddings, df):
    np.save(INDEX_FILE, embeddings)
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            "title": row["title"],
            "department": row["department"],
            "category": row["category"],
            "policy_text": row["policy_text"]
        })
    with open(INDEX_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_vector_index():
    if not Path(INDEX_FILE).exists():
        return None
    try:
        return np.load(INDEX_FILE)
    except Exception:
        return None


def embed_question(question):
    result = genai.embed_content(model=EMBEDDING_MODEL, content=question)
    return np.array(result["embedding"], dtype=np.float32)


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0
    return np.dot(a, b) / denom


def retrieve_policy(question, df, embeddings):
    query_embedding = embed_question(question)
    similarities = [cosine_similarity(query_embedding, p_emb) for p_emb in embeddings]
    best_idx = int(np.argmax(similarities))
    return df.iloc[best_idx], similarities[best_idx]


def llm_with_rag(question, df, model, embeddings):
    start_time = time.perf_counter()
    try:
        policy, similarity = retrieve_policy(question, df, embeddings)
        evidence = f"Title: {policy['title']}\nDepartment: {policy['department']}\nCategory: {policy['category']}\nPolicy text: {policy['policy_text']}"

        prompt = f"""
You are a company policy assistant.
Answer the employee's question using ONLY the policy evidence provided below.
Do not invent information.

Employee question:
{question}

Retrieved policy evidence:
{evidence}

At the END of your response, include exactly:
POLICY: {policy['title']}
SUPPORT: [Supported / Not supported / Unclear]
"""
        response = model.generate_content(prompt)
        text = response.text
        parsed = parse_llm_response(text)
        elapsed = time.perf_counter() - start_time
        estimated_tokens = max(1, len(text.split()) * 4 // 3)

        return {
            "answer": parsed["answer"],
            "policy_title": policy["title"],
            "policy": policy,
            "time": elapsed,
            "tokens": estimated_tokens,
            "support": parsed["support"],
            "similarity": similarity
        }
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return {
            "answer": f"RAG error: {str(e)}",
            "policy_title": "Error",
            "policy": None,
            "time": elapsed,
            "tokens": 0,
            "support": "Unavailable",
            "similarity": 0
        }


def get_or_create_vector_index(df):
    embeddings = load_vector_index()
    if embeddings is not None and embeddings.shape[0] == len(df):
        return embeddings
    try:
        with st.spinner("Creating vector index for policies..."):
            embeddings = create_embeddings(df)
            save_vector_index(embeddings, df)
        return embeddings
    except Exception as e:
        st.error(f"Could not create vector index: {str(e)}")
        return None


# ============================================================
# DISPLAY HELPERS
# ============================================================

def display_policy(policy):
    if policy is None:
        st.info("No specific policy identified.")
        return
    st.markdown(
        f"""
        <div class="policy-box">
        <strong>{policy['title']}</strong><br>
        <small>Department: {policy['department']} | Category: {policy['category']}</small>
        <br><br>
        {policy['policy_text']}
        </div>
        """,
        unsafe_allow_html=True
    )


def display_result(result, method_name):
    st.markdown(f'<div class="method-header">{method_name}</div>', unsafe_allow_html=True)
    st.markdown("**Answer**")
    st.write(result["answer"])
    st.markdown("**Relevant policy**")
    if result.get("policy") is not None:
        display_policy(result["policy"])
    else:
        st.write(result.get("policy_title", "None"))

    st.markdown("**Performance**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Response time", f"{result['time']:.3f} s")
    with col2:
        st.metric("Estimated tokens", f"{result['tokens']:,}")

    st.markdown("**Database support**")
    support = result.get("support", "Unknown")
    if "supported" in support.lower():
        st.success(support)
    elif "not supported" in support.lower():
        st.error(support)
    else:
        st.warning(support)

    if "similarity" in result:
        st.caption(f"RAG cosine similarity: {result['similarity']:.4f}")


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    st.markdown('<div class="main-title">📋 Company Policy Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Comparison of rules-based search, LLM without retrieval, and RAG.</div>',
        unsafe_allow_html=True
    )

    try:
        df = load_policies()
    except Exception as e:
        st.error(f"Could not load policy database: {e}")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        st.write(f"**Policies loaded:** {len(df)}")
        st.write(f"**Categories:** {df['category'].nunique()}")
        st.write(f"**Departments:** {df['department'].nunique()}")
        st.divider()
        st.subheader("Gemini")
        st.write(f"Generation model: `{GENERATION_MODEL}`")
        st.write(f"Embedding model: `{EMBEDDING_MODEL}`")

        api_key = get_api_key()
        if api_key:
            st.success("Gemini API key detected")
        else:
            st.error("Gemini API key not detected")

        st.divider()
        if st.button("Rebuild vector index"):
            try:
                with st.spinner("Rebuilding embeddings..."):
                    embeddings = create_embeddings(df)
                    save_vector_index(embeddings, df)
                st.success("Vector index rebuilt successfully.")
            except Exception as e:
                st.error(f"Could not rebuild index: {e}")

    st.subheader("Ask a policy question")
    question = st.text_input(
        "Example: How often are employee performance reviews conducted?",
        placeholder="Type your company policy question here..."
    )

    compare_button = st.button("🔎 Compare all three methods", type="primary")

    if not compare_button:
        st.info("Enter a question and click 'Compare all three methods'.")
        return

    if not question.strip():
        st.warning("Please enter a question first.")
        return

    model, api_error = initialize_gemini()
    rules_result = rules_based_search(question, df)

    if model is None:
        llm_result = {"answer": api_error, "policy_title": "Unavailable", "policy": None, "time": 0, "tokens": 0, "support": "Unavailable"}
        rag_result = llm_result.copy()
    else:
        with st.spinner("Running LLM without vector retrieval..."):
            llm_result = llm_without_rag(question, df, model)

        embeddings = get_or_create_vector_index(df)
        if embeddings is None:
            rag_result = {"answer": "Vector index error.", "policy_title": "Unavailable", "policy": None, "time": 0, "tokens": 0, "support": "Unavailable", "similarity": 0}
        else:
            with st.spinner("Running Gemini + RAG..."):
                rag_result = llm_with_rag(question, df, model, embeddings)

    st.divider()
    st.subheader("Method comparison")
    col1, col2, col3 = st.columns(3)
    with col1:
        display_result(rules_result, "1️⃣ Rules-based search")
    with col2:
        display_result(llm_result, "2️⃣ LLM without vector index")
    with col3:
        display_result(rag_result, "3️⃣ LLM + Vector Index (RAG)")


if __name__ == "__main__":
    main()
