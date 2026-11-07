import html
from pathlib import Path

import streamlit as st

from src.model_utils import (
    load_model,
    correct_paragraph,
    get_text_level_changes,
    render_word_highlights_html,
)


# ---------------------------------------------------------
# Page setup and project assets
# ---------------------------------------------------------

st.set_page_config(
    page_title="GrammarAI",
    page_icon="✍️",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
ASSET_DIR = PROJECT_DIR / "assets"
MASCOT_PATH = ASSET_DIR / "grammarai_mascot_quill.png"


# ---------------------------------------------------------
# Place the supplied decorative frame around the app
# ---------------------------------------------------------

st.markdown(
    f"""
    <style>
    /* Light workspace behind the transparent frame */
    .stApp {{
        background: #F7F9FD;
        color: #243047;
    }}

    /* Keep the app content inside the frame's inner area. */
    .block-container {{
        width: 94%;
        max-width: 1540px;
        padding-top: 3.2rem;
        padding-bottom: 3.5rem;
        padding-left: 3.6rem;
        padding-right: 3.6rem;
    }}

    h1, h2, h3, p, label, .stMarkdown {{
        color: #243047;
    }}

    .ga-header {{
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 5px 0 15px;
        margin-bottom: 16px;
        border-bottom: 1px solid #DFE6F1;
    }}

    .ga-brand {{
        font-size: 29px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: #202B42;
        line-height: 1.1;
    }}

    .ga-brand span {{
        color: #4776D0;
    }}

    .ga-tagline {{
        color: #6C7890;
        font-size: 13px;
        margin-top: 4px;
    }}

    .ga-intro {{
        text-align: center;
        padding: 7px 0 17px;
    }}

    .ga-intro h2 {{
        color: #25334D;
        font-size: clamp(25px, 3vw, 36px);
        font-weight: 800;
        margin: 0 0 5px;
    }}

    .ga-intro p {{
        color: #69768D;
        font-size: 14px;
        margin: 0;
    }}

    .ga-panel-title {{
        font-size: 17px;
        font-weight: 750;
        color: #26344D;
        margin: 0 0 8px;
    }}

    .ga-result-box {{
        min-height: 190px;
        padding: 16px 18px;
        border: 1px solid #DCE4F0;
        border-radius: 12px;
        background: rgba(255,255,255,.96);
        color: #26344D;
        line-height: 1.85;
        font-size: 15px;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }}

    .ga-result-box .grammarai-highlight-container {{
        background: transparent !important;
        color: #26344D !important;
        font-family: inherit !important;
        line-height: 1.85 !important;
        white-space: pre-wrap !important;
    }}

    .ga-result-box .grammarai-changed-word,
    .ga-result-box .grammarai-punctuation-insertion {{
        color: #C6283D !important;
        text-decoration-color: #C6283D !important;
    }}

    .ga-suggestions {{
        margin-top: 17px;
        padding: 16px 19px;
        background: rgba(255,255,255,.97);
        border: 1px solid #DCE4F0;
        border-radius: 12px;
    }}

    .ga-suggestions-title {{
        color: #26344D;
        font-weight: 800;
        font-size: 18px;
        margin-bottom: 4px;
    }}

    .ga-suggestions-subtitle {{
        color: #758198;
        font-size: 12px;
        margin-bottom: 9px;
    }}

    .ga-suggestion-line {{
        padding: 8px 0;
        border-top: 1px solid #EDF1F6;
        color: #34425A;
        font-size: 14px;
    }}

    .stTextArea textarea {{
        background: #FFFFFF !important;
        color: #243047 !important;
        border: 1px solid #D4DDEA !important;
        border-radius: 12px !important;
        font-size: 15px !important;
        line-height: 1.65 !important;
    }}

    .stTextArea textarea:focus {{
        border-color: #7095D7 !important;
        box-shadow: 0 0 0 2px rgba(71,118,208,.13) !important;
    }}

    div.stButton > button[kind="primary"] {{
        background: #356CC4 !important;
        color: #FFFFFF !important;
        border: 1px solid #356CC4 !important;
        border-radius: 9px !important;
        min-height: 44px;
        font-size: 15px;
        font-weight: 750;
    }}

    div.stButton > button[kind="primary"]:hover {{
        background: #285AA9 !important;
        border-color: #285AA9 !important;
        color: #FFFFFF !important;
    }}

    div.stButton > button:not([kind="primary"]) {{
        background: #FFFFFF !important;
        color: #34425A !important;
        border: 1px solid #D4DDEA !important;
        border-radius: 9px !important;
    }}

    @media (max-width: 850px) {{
        .block-container {{
            width: 96%;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
            padding-top: 1rem;
        }}

        .ga-brand {{
            font-size: 24px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load the existing GrammarAI model
# ---------------------------------------------------------

@st.cache_resource
def load_grammar_model():
    # Reuse the pretrained Vennify T5 model on CPU.
    tokenizer, model, device = load_model(device="cpu")
    return tokenizer, model


tokenizer, model = load_grammar_model()


# ---------------------------------------------------------
# Simple branded header and input
# ---------------------------------------------------------

header_image_col, header_text_col = st.columns([0.75, 5], vertical_alignment="center")

with header_image_col:
    if MASCOT_PATH.is_file():
        st.image(str(MASCOT_PATH), width=112)

with header_text_col:
    st.markdown(
        """
        <div class="ga-header">
            <div>
                <div class="ga-brand">Grammar<span>AI</span></div>
                <div class="ga-tagline">Your intelligent grammar assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="ga-intro">
        <h2>Make every sentence clearer.</h2>
        <p>Check grammar, identify possible errors, and see the corrected text.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

text = st.text_area(
    "Enter your text:",
    placeholder="Example: She go to school every day.",
    height=155,
)

# ---------------------------------------------------------
# Run grammar correction
# ---------------------------------------------------------

if st.button("Check Grammar", type="primary", use_container_width=True):

    if not text.strip():
        st.warning("Please enter some text.")

    else:
        try:
            with st.spinner("Checking your text..."):

                # 1. Correct the input using the existing T5 model.
                corrected_text = correct_paragraph(
                    text,
                    tokenizer,
                    model,
                )

                # 2. Compare original and corrected text.
                changes = get_text_level_changes(
                    text,
                    corrected_text,
                )

                # 3. Highlight changed words and punctuation in the original.
                highlighted_text = render_word_highlights_html(
                    text,
                    corrected_text,
                )

            st.subheader("Grammar correction results")

            # Display the original and corrected text side by side.
            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown(
                    '<div class="ga-panel-title">Original text — changes highlighted in red</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="ga-result-box">{highlighted_text}</div>',
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    '<div class="ga-panel-title">Corrected text</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="ga-result-box">{html.escape(corrected_text)}</div>',
                    unsafe_allow_html=True,
                )

            # Display a short, read-only list of suggested changes.
            suggestion_rows = []

            for index, change in enumerate(changes, start=1):
                original_part = str(change.get("original", ""))
                suggestion = str(change.get("suggestion", ""))
                kind = change.get("kind", "word")
                change_type = change.get("change_type", "replace")
                position = change.get("position", "before")

                if kind == "word_insertion":
                    description = (
                        f"Add '{suggestion}' {position} '{original_part}'"
                    )
                elif kind == "punctuation_insertion":
                    description = f"Add punctuation '{suggestion}'"
                elif change_type == "delete":
                    description = f"Remove '{original_part}'"
                elif kind == "punctuation":
                    description = (
                        f"Change punctuation '{original_part}' to '{suggestion}'"
                    )
                else:
                    description = (
                        f"Change '{original_part}' to '{suggestion}'"
                    )

                suggestion_rows.append(
                    f'<div class="ga-suggestion-line">'
                    f'<strong>{index}.</strong> {html.escape(description)}'
                    f'</div>'
                )

            if not changes:
                suggestion_rows.append(
                    '<div class="ga-suggestion-line">'
                    'No changes were suggested by the model.'
                    '</div>'
                )

            # Render the entire panel in one Markdown call so Streamlit does
            # not parse an opening/closing div as separate Markdown blocks.
            suggestions_html = (
                '<div class="ga-suggestions">'
                f'<div class="ga-suggestions-title">Suggested corrections ({len(changes)})</div>'
                '<div class="ga-suggestions-subtitle">'
                'Changes identified by comparing the original text with the model output.'
                '</div>'
                f'{"".join(suggestion_rows)}'
                '</div>'
            )
            st.markdown(suggestions_html, unsafe_allow_html=True)

            st.caption(
                "Red underlines indicate changes suggested by the model. "
                "They are potential corrections, not independently verified errors."
            )

        except Exception as error:
            st.error(f"An error occurred: {error}")
