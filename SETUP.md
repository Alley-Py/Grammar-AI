# GrammarAI — Light Theme Streamlit App

This version uses a restrained light theme: white surfaces, light grey backgrounds,
blue primary controls, and red underlines for detected model changes.

## Existing project layout

Copy `app.py` into the root of your existing project, alongside `README.md`:

```text
GrammarAI_Project/
├── app.py
├── assets/
│   └── grammarai_mascot_quill.png
├── src/
│   ├── __init__.py
│   ├── model_utils.py
│   └── correction_manager.py
└── ...
```

Do not replace your existing backend modules with this starter package.

## Install Streamlit

Activate the environment where PyTorch and Transformers already work:

```bash
conda activate local_gpu_env
python -m pip install "streamlit>=1.38,<2.0"
```

## Launch

From the project root (the folder containing `app.py`):

```bash
python -m streamlit run app.py
```

The app expects these names in `src/model_utils.py`:
- `load_model`
- `correct_paragraph`
- `get_text_level_changes`
- `render_word_highlights_html`
- `validate_output`

And these names in `src/correction_manager.py`:
- `CorrectionManager`
- `CorrectionConflictError`

The model loads once per running Streamlit process through `st.cache_resource`.

## Scope

This app implements grammar correction only. It does not provide professional
rewriting. Red underlines show differences between original text and model output;
they are candidate suggestions, not independently verified error classifications.
