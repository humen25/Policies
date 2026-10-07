import os
import re
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Company Policy Assistant",
    page_icon="📋",
    layout="wide"
)

CSV_FILE = "company_policies.csv"

GENERATION_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

EMBEDDING_MODEL = os.environ.get(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001"
)

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

    .policy-box {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 12px;
        background-color: #fafafa;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API KEY
# ============================================================

def get_api_key():

    # Streamlit secrets
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # Environment variable
    return os.environ.get("GEMINI_API_KEY")


def initialize_gemini():

    api_key = get_api_key()

    if not api_key:
        return None, (
            "Gemini API key not found. "
            "Set GEMINI_API_KEY as an environment variable "
            "or in Streamlit secrets."
        )

    try:

        client = genai.Client(
            api_key=api_key
        )

        return client, None

    except Exception as e:

        return None, (
            f"Could not initialize Gemini: {str(e)}"
        )


# ============================================================
# LOAD POLICIES
# ============================================================

@st.cache_data
def load_policies():

    if not Path(CSV_FILE).exists():

        raise FileNotFoundError(
            f"Could not find {CSV_FILE}. "
            "Make sure it is in the same folder as app.py."
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
            f"CSV is missing these columns: {missing}"
        )

    for col in required_columns:

        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
        )

    return df


# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text):

    return set(
        normalize_text(text).split()
    )


# ============================================================
# 1. RULES-BASED SEARCH
# ============================================================

def rules_based_search(question, df):

    start_time = time.perf_counter()

    question_tokens = tokenize(question)

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "what",
        "how",
        "can",
        "i",
        "do",
        "does",
        "of",
        "to",
        "for",
        "and",
        "or",
        "in",
        "on",
        "my",
        "our",
        "company",
        "policy",
        "about",
        "please",
        "tell",
        "me"
    }

    question_tokens = {
        word
        for word in question_tokens
        if word not in stopwords
        and len(word) > 2
    }

    scores = []

    for _, row in df.iterrows():

        searchable_text = " ".join([
            row["title"],
            row["department"],
            row["category"],
            row["policy_text"]
        ])

        policy_tokens = tokenize(
            searchable_text
        )

        overlap = (
            question_tokens
            .intersection(policy_tokens)
        )

        score = len(overlap)

        # Give title matches extra weight
        title_tokens = tokenize(
            row["title"]
        )

        title_overlap = (
            question_tokens
            .intersection(title_tokens)
        )

        score += len(title_overlap) * 3

        scores.append(score)

    df_temp = df.copy()

    df_temp["rule_score"] = scores

    best_idx = df_temp["rule_score"].idxmax()

    best_score = df_temp.loc[
        best_idx,
        "rule_score"
    ]

    elapsed = (
        time.perf_counter()
        - start_time
    )

    if best_score == 0:

        return {
            "answer": (
                "No clear policy match was "
                "found using keyword rules."
            ),
            "policy": None,
            "time": elapsed,
            "tokens": 0,
            "support": "Not supported / no match",
            "score": 0
        }

    policy = df_temp.loc[best_idx]

    answer = (
        f"According to the **{policy['title']}**, "
        f"{policy['policy_text']}"
    )

    estimated_tokens = max(
        1,
        len(answer.split()) * 4 // 3
    )

    return {
        "answer": answer,
        "policy": policy,
        "time": elapsed,
        "tokens": estimated_tokens,
        "support": "Supported by database",
        "score": best_score
    }


# ============================================================
# RAW POLICY DATABASE FOR LLM WITHOUT RAG
# ============================================================

def build_raw_policy_context(
    df,
    max_chars=80000
):

    sections = []

    for i, row in df.iterrows():

        section = f"""
POLICY {i + 1}

Title: {row['title']}

Department: {row['department']}

Category: {row['category']}

Policy text:
{row['policy_text']}
"""

        sections.append(section)

    context = "\n".join(sections)

    if len(context) > max_chars:

        context = context[:max_chars]

        context += (
            "\n[Policy database truncated "
            "because of prompt size.]"
        )

    return context


# ============================================================
# FIND POLICY BY TITLE
# ============================================================

def find_policy_by_title(
    title,
    df
):

    if not title:
        return None

    title_normalized = normalize_text(
        title
    )

    # Exact match
    for _, row in df.iterrows():

        if (
            normalize_text(row["title"])
            == title_normalized
        ):

            return row

    # Partial match
    for _, row in df.iterrows():

        normalized = normalize_text(
            row["title"]
        )

        if (
            title_normalized in normalized
            or normalized in title_normalized
        ):

            return row

    return None


# ============================================================
# TOKEN INFORMATION
# ============================================================

def get_token_usage(response):

    """
    Try to obtain actual Gemini token usage.

    If unavailable, return an estimate.
    """

    try:

        usage = response.usage_metadata

        input_tokens = (
            getattr(
                usage,
                "prompt_token_count",
                None
            )
            or getattr(
                usage,
                "prompt_token_count",
                0
            )
        )

        output_tokens = (
            getattr(
                usage,
                "candidates_token_count",
                None
            )
            or getattr(
                usage,
                "candidates_token_count",
                0
            )
        )

        total_tokens = (
            getattr(
                usage,
                "total_token_count",
                None
            )
            or (
                input_tokens
                + output_tokens
            )
        )

        return {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens
        }

    except Exception:

        text = getattr(
            response,
            "text",
            ""
        )

        estimated = max(
            1,
            len(text.split()) * 4 // 3
        )

        return {
            "input": 0,
            "output": estimated,
            "total": estimated
        }


# ============================================================
# PARSE LLM RESPONSE
# ============================================================

def parse_llm_response(text):

    result = {
        "answer": text,
        "policy_title": "Not identified",
        "support": "Unclear"
    }

    policy_match = re.search(
        r"POLICY\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE
    )

    if policy_match:

        result["policy_title"] = (
            policy_match.group(1).strip()
        )

    support_match = re.search(
        r"SUPPORT\s*:\s*(.+)",
        text,
        flags=re.IGNORECASE
    )

    if support_match:

        result["support"] = (
            support_match.group(1).strip()
        )

    # Remove metadata lines
    cleaned = re.sub(
        r"^\s*POLICY\s*:.*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    cleaned = re.sub(
        r"^\s*SUPPORT\s*:.*$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE
    )

    result["answer"] = cleaned.strip()

    return result


# ============================================================
# 2. LLM WITHOUT VECTOR INDEX
# ============================================================

def llm_without_rag(
    question,
    df,
    client
):

    start_time = time.perf_counter()

    context = build_raw_policy_context(
        df
    )

    prompt = f"""
You are a company policy assistant.

You have access to the company's complete
policy database below.

Answer the employee's question using ONLY
information contained in the policy database.

Do not invent policies, rules, dates,
requirements, or exceptions.

If the database does not contain enough
information to answer the question, say so.

Identify the policy that is most relevant.

EMPLOYEE QUESTION:
{question}

COMPANY POLICY DATABASE:
{context}

At the END of your response, include exactly:

POLICY: [most relevant policy title]
SUPPORT: [Supported / Not supported / Unclear]
"""

    try:

        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )

        text = response.text

        parsed = parse_llm_response(
            text
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        usage = get_token_usage(
            response
        )

        return {
            "answer": parsed["answer"],
            "policy_title": parsed["policy_title"],
            "policy": find_policy_by_title(
                parsed["policy_title"],
                df
            ),
            "time": elapsed,
            "tokens": usage["total"],
            "input_tokens": usage["input"],
            "output_tokens": usage["output"],
            "support": parsed["support"]
        }

    except Exception as e:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "answer": (
                f"Gemini API error: {str(e)}"
            ),
            "policy_title": "Error",
            "policy": None,
            "time": elapsed,
            "tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "support": "Unavailable"
        }


# ============================================================
# CREATE VECTOR EMBEDDINGS
# ============================================================

def create_embeddings(
    df,
    client
):

    texts = []

    for _, row in df.iterrows():

        text = f"""
Title: {row['title']}

Department: {row['department']}

Category: {row['category']}

Policy:
{row['policy_text']}
"""

        texts.append(text)

    # Gemini supports embedding multiple strings
    # in one request for gemini-embedding-001.
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    embeddings = []

    for embedding in result.embeddings:

        embeddings.append(
            embedding.values
        )

    return np.array(
        embeddings,
        dtype=np.float32
    )


# ============================================================
# SAVE VECTOR INDEX
# ============================================================

def save_vector_index(
    embeddings,
    df
):

    np.save(
        INDEX_FILE,
        embeddings
    )

    metadata = []

    for _, row in df.iterrows():

        metadata.append({
            "title": row["title"],
            "department": row["department"],
            "category": row["category"],
            "policy_text": row["policy_text"]
        })

    with open(
        INDEX_METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# LOAD VECTOR INDEX
# ============================================================

def load_vector_index():

    if not Path(INDEX_FILE).exists():

        return None

    try:

        return np.load(
            INDEX_FILE
        )

    except Exception:

        return None


# ============================================================
# EMBED USER QUESTION
# ============================================================

def embed_question(
    question,
    client
):

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    return np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a,
    b
):

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:

        return 0

    return (
        np.dot(a, b)
        / denominator
    )


# ============================================================
# RETRIEVE POLICY
# ============================================================

def retrieve_policy(
    question,
    df,
    embeddings,
    client
):

    query_embedding = embed_question(
        question,
        client
    )

    similarities = []

    for policy_embedding in embeddings:

        similarity = cosine_similarity(
            query_embedding,
            policy_embedding
        )

        similarities.append(
            similarity
        )

    best_idx = int(
        np.argmax(similarities)
    )

    return (
        df.iloc[best_idx],
        similarities[best_idx]
    )


# ============================================================
# 3. LLM + RAG
# ============================================================

def llm_with_rag(
    question,
    df,
    client,
    embeddings
):

    start_time = time.perf_counter()

    try:

        policy, similarity = (
            retrieve_policy(
                question,
                df,
                embeddings,
                client
            )
        )

        evidence = f"""
Title: {policy['title']}

Department: {policy['department']}

Category: {policy['category']}

Policy text:
{policy['policy_text']}
"""

        prompt = f"""
You are a company policy assistant.

Answer the employee's question using ONLY
the retrieved policy evidence below.

Do not invent information.

If the retrieved policy does not contain
enough information, say that clearly.

EMPLOYEE QUESTION:
{question}

RETRIEVED POLICY EVIDENCE:
{evidence}

At the END of your response, include exactly:

POLICY: {policy['title']}
SUPPORT: [Supported / Not supported / Unclear]
"""

        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )

        text = response.text

        parsed = parse_llm_response(
            text
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        usage = get_token_usage(
            response
        )

        return {
            "answer": parsed["answer"],
            "policy_title": policy["title"],
            "policy": policy,
            "time": elapsed,
            "tokens": usage["total"],
            "input_tokens": usage["input"],
            "output_tokens": usage["output"],
            "support": parsed["support"],
            "similarity": similarity
        }

    except Exception as e:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "answer": (
                f"RAG error: {str(e)}"
            ),
            "policy_title": "Error",
            "policy": None,
            "time": elapsed,
            "tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "support": "Unavailable",
            "similarity": 0
        }


# ============================================================
# VECTOR INDEX MANAGEMENT
# ============================================================

def get_or_create_vector_index(
    df,
    client
):

    embeddings = load_vector_index()

    if (
        embeddings is not None
        and embeddings.shape[0] == len(df)
    ):

        return embeddings

    try:

        with st.spinner(
            "Creating vector index for the 98 policies..."
        ):

            embeddings = create_embeddings(
                df,
                client
            )

            save_vector_index(
                embeddings,
                df
            )

        return embeddings

    except Exception as e:

        st.error(
            f"Could not create vector index: {str(e)}"
        )

        return None


# ============================================================
# DISPLAY POLICY
# ============================================================

def display_policy(policy):

    if policy is None:

        st.info(
            "No specific policy identified."
        )

        return

    st.markdown(
        f"""
        <div class="policy-box">

        <strong>{policy['title']}</strong><br>

        <small>
        Department: {policy['department']} |
        Category: {policy['category']}
        </small>

        <br><br>

        {policy['policy_text']}

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    result,
    method_name
):

    st.markdown(
        f'<div class="method-header">{method_name}</div>',
        unsafe_allow_html=True
    )

    st.markdown("**Answer**")

    st.write(
        result["answer"]
    )

    st.markdown(
        "**Relevant policy**"
    )

    if result.get("policy") is not None:

        display_policy(
            result["policy"]
        )

    else:

        st.write(
            result.get(
                "policy_title",
                "None"
            )
        )

    st.markdown(
        "**Performance**"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Response time",
            f"{result['time']:.3f} s"
        )

    with col2:

        st.metric(
            "Tokens",
            f"{result['tokens']:,}"
        )

    # Show actual input/output tokens
    if (
        result.get("input_tokens", 0)
        or result.get("output_tokens", 0)
    ):

        st.caption(
            f"Input: {result.get('input_tokens', 0):,} "
            f"| Output: {result.get('output_tokens', 0):,}"
        )

    st.markdown(
        "**Database support**"
    )

    support = result.get(
        "support",
        "Unknown"
    )

    if (
        "supported" in support.lower()
        and "not supported"
        not in support.lower()
    ):

        st.success(
            support
        )

    elif (
        "not supported"
        in support.lower()
    ):

        st.error(
            support
        )

    else:

        st.warning(
            support
        )

    if "similarity" in result:

        st.caption(
            f"RAG cosine similarity: "
            f"{result['similarity']:.4f}"
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    st.markdown(
        '<div class="main-title">'
        '📋 Company Policy Assistant'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Comparison of rules-based search, '
        'LLM without retrieval, and '
        'Retrieval-Augmented Generation (RAG).'
        '</div>',
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    try:

        df = load_policies()

    except Exception as e:

        st.error(
            f"Could not load policy database: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.header(
            "Settings"
        )

        st.write(
            f"**Policies loaded:** {len(df)}"
        )

        st.write(
            f"**Categories:** "
            f"{df['category'].nunique()}"
        )

        st.write(
            f"**Departments:** "
            f"{df['department'].nunique()}"
        )

        st.divider()

        st.subheader(
            "Gemini"
        )

        st.write(
            f"Generation model: "
            f"`{GENERATION_MODEL}`"
        )

        st.write(
            f"Embedding model: "
            f"`{EMBEDDING_MODEL}`"
        )

        api_key = get_api_key()

        if api_key:

            st.success(
                "Gemini API key detected"
            )

        else:

            st.error(
                "Gemini API key not detected"
            )

        st.divider()

        st.subheader(
            "Vector index"
        )

        if Path(INDEX_FILE).exists():

            st.success(
                "Vector index found"
            )

        else:

            st.info(
                "No vector index found yet."
            )

        if st.button(
            "Rebuild vector index"
        ):

            client, error = (
                initialize_gemini()
            )

            if client is None:

                st.error(
                    error
                )

            else:

                try:

                    with st.spinner(
                        "Rebuilding embeddings..."
                    ):

                        embeddings = (
                            create_embeddings(
                                df,
                                client
                            )
                        )

                        save_vector_index(
                            embeddings,
                            df
                        )

                    st.success(
                        "Vector index rebuilt successfully."
                    )

                except Exception as e:

                    st.error(
                        f"Could not rebuild index: {e}"
                    )

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    st.subheader(
        "Ask a policy question"
    )

    question = st.text_input(
        "Example: How often are employee performance reviews conducted?",
        placeholder=(
            "Type your company policy question here..."
        )
    )

    compare_button = st.button(
        "🔎 Compare all three methods",
        type="primary"
    )

    if not compare_button:

        st.info(
            "Enter a question and click "
            "'Compare all three methods'."
        )

        return

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

        return

    # --------------------------------------------------------
    # GEMINI CLIENT
    # --------------------------------------------------------

    client, api_error = (
        initialize_gemini()
    )

    # --------------------------------------------------------
    # METHOD 1
    # --------------------------------------------------------

    rules_result = (
        rules_based_search(
            question,
            df
        )
    )

    # --------------------------------------------------------
    # METHODS 2 + 3
    # --------------------------------------------------------

    if client is None:

        llm_result = {
            "answer": api_error,
            "policy_title": "Unavailable",
            "policy": None,
            "time": 0,
            "tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "support": "Unavailable"
        }

        rag_result = {
            **llm_result,
            "similarity": 0
        }

    else:

        # ----------------------------------------------------
        # METHOD 2
        # ----------------------------------------------------

        with st.spinner(
            "Running LLM without vector retrieval..."
        ):

            llm_result = (
                llm_without_rag(
                    question,
                    df,
                    client
                )
            )

        # ----------------------------------------------------
        # METHOD 3
        # ----------------------------------------------------

        embeddings = (
            get_or_create_vector_index(
                df,
                client
            )
        )

        if embeddings is None:

            rag_result = {
                "answer": (
                    "Vector index could not "
                    "be created."
                ),
                "policy_title": "Unavailable",
                "policy": None,
                "time": 0,
                "tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "support": "Unavailable",
                "similarity": 0
            }

        else:

            with st.spinner(
                "Running Gemini + RAG..."
            ):

                rag_result = (
                    llm_with_rag(
                        question,
                        df,
                        client,
                        embeddings
                    )
                )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Method comparison"
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        display_result(
            rules_result,
            "1️⃣ Rules-based search"
        )

    with col2:

        display_result(
            llm_result,
            "2️⃣ LLM without vector index"
        )

    with col3:

        display_result(
            rag_result,
            "3️⃣ LLM + Vector Index (RAG)"
        )

    # --------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📊 Comparison summary"
    )

    summary = pd.DataFrame({

        "Method": [
            "Rules-based",
            "LLM without vector index",
            "LLM + RAG"
        ],

        "Response time (s)": [
            round(
                rules_result["time"],
                3
            ),
            round(
                llm_result["time"],
                3
            ),
            round(
                rag_result["time"],
                3
            )
        ],

        "Tokens": [
            rules_result["tokens"],
            llm_result["tokens"],
            rag_result["tokens"]
        ],

        "Relevant policy": [

            (
                rules_result["policy"]["title"]
                if rules_result["policy"]
                is not None
                else "None"
            ),

            llm_result[
                "policy_title"
            ],

            rag_result[
                "policy_title"
            ]
        ],

        "Database support": [
            rules_result["support"],
            llm_result["support"],
            rag_result["support"]
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
