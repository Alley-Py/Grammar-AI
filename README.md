<p align="center">
  <img src="assets/grammarai_mascot_quill.png" alt="GrammarAI mascot" width="150">
</p>

<h1 align="center">🪶 GrammarAI</h1>

<p align="center">
  <strong>Pretrained Transformer-Based Grammar Correction Application</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-Inference-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/Model-T5--Base-2563EB" alt="T5 Base">
  <img src="https://img.shields.io/badge/Streamlit-Application-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/PySBD-Sentence%20Segmentation-6C5CE7" alt="PySBD">
</p>

<p align="center">
  <a href="#features"><img src="https://img.shields.io/badge/Explore-Features-2563EB?style=for-the-badge" alt="Explore Features"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Quick-Setup-16865B?style=for-the-badge" alt="Quick Setup"></a>
  <a href="GrammarAI_CRISP_MLQ_Project_Presentation.pptx"><img src="https://img.shields.io/badge/Project-Presentation-8B5CF6?style=for-the-badge" alt="Project Presentation"></a>
  <a href="GrammarAI_development.ipynb"><img src="https://img.shields.io/badge/Development-Notebook-DB8B22?style=for-the-badge" alt="Development Notebook"></a>
  <a href="https://huggingface.co/vennify/t5-base-grammar-correction"><img src="https://img.shields.io/badge/Pretrained-Model-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Pretrained Model"></a>
</p>

---

## 📌 Overview

**GrammarAI** is an NLP-powered grammar correction application that uses a pretrained **T5 (Text-to-Text Transfer Transformer)** model to identify and correct grammatical issues in sentences and paragraphs.

The application integrates the pretrained [`vennify/t5-base-grammar-correction`](https://huggingface.co/vennify/t5-base-grammar-correction) checkpoint with **PyTorch**, **Hugging Face Transformers**, and **Streamlit**. It displays the original text with model-suggested changes highlighted in red, the corrected text alongside it, and a concise list explaining the suggested changes.

> **Model-training scope:** This project uses the pretrained T5 checkpoint for inference. The model was not retrained or fine-tuned as part of this project.

## ✨ Features

- **Grammar correction:** Corrects grammar in user-provided sentences and paragraphs.
- **Pretrained T5 inference:** Uses an existing encoder-decoder Transformer checkpoint without project-specific training.
- **Sentence-level processing:** Uses PySBD sentence segmentation as part of paragraph processing.
- **Red-highlighted changes:** Highlights differences between the original and model-corrected text, including word and punctuation changes.
- **Side-by-side results:** Displays the original input and corrected output together.
- **Suggested corrections:** Presents a numbered, read-only list explaining detected replacements, insertions, deletions, and punctuation changes.
- **Lightweight deployment:** Provides a simple Streamlit interface and supports CPU inference; a GPU is not required for the current local workflow.

## 🖥️ Application Preview

The application follows a simple workflow:

1. Enter a sentence or paragraph.
2. Select **Check Grammar**.
3. Review the original text with red highlights.
4. Compare it with the corrected output.
5. Read the numbered suggested-corrections list.

<!-- Add a current application screenshot to assets/screenshots/ and uncomment:
<p align="center">
  <img src="assets/screenshots/grammarai-app.png" alt="GrammarAI application preview" width="900">
</p>
-->

## 🧠 Model and Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Grammar correction model | `vennify/t5-base-grammar-correction` | Generates corrected text |
| Model architecture | T5 / `T5ForConditionalGeneration` | Sequence-to-sequence text generation |
| Deep learning framework | PyTorch | Loads the model and performs inference |
| Model integration | Hugging Face Transformers | Loads the pretrained model and tokenizer |
| Sentence segmentation | PySBD | Segments paragraph text into sentences |
| User interface | Streamlit | Provides the interactive grammar-checking interface |
| Text comparison | Project utility functions | Identifies differences between original and corrected text |

### Why a pretrained T5 model?

T5 is an encoder-decoder Transformer designed for text-to-text tasks. Grammar correction can be treated as a text transformation task: the model receives an input sentence and generates a corrected version.

This project uses the pretrained grammar-correction checkpoint from Hugging Face. Reusing the checkpoint allowed the application to focus on inference, text comparison, highlighting, and deployment rather than training a new model.

**Model checkpoint:** [`vennify/t5-base-grammar-correction`](https://huggingface.co/vennify/t5-base-grammar-correction)

## 🔄 Application Workflow

```text
             User Input
                  |
                  v
        Input Validation
                  |
                  v
       Paragraph Processing
          (PySBD)
                  |
                  v
        T5 Tokenization
                  |
                  v
      Pretrained T5 Model
          (PyTorch)
                  |
                  v
       Corrected Text Output
                  |
          +-------+-------+
          |               |
          v               v
   Text Comparison   Original vs.
   and Alignment     Corrected Text
          |               |
          v               v
   Suggested Changes  Red Highlights
          |               |
          +-------+-------+
                  |
                  v
       Streamlit Results View
```

### Output components

| Output | Description |
|---|---|
| Original text | User-provided text with detected differences highlighted in red |
| Corrected text | Complete text generated by the pretrained T5 model |
| Suggested corrections | Read-only descriptions of detected word replacements, insertions, deletions, and punctuation changes |

**Important:** Red highlights and the suggested-corrections list represent differences between the original input and the model output. They are model-suggested edits, not independently verified grammatical-error labels. The generated output should be reviewed before use.

## 🧪 Project Methodology — CRISP-ML(Q)

The project follows an adapted **CRISP-ML(Q)** lifecycle. Since the application reuses a pretrained model, the workflow focuses on model selection, inference-time text preparation, evaluation, deployment, and quality assurance rather than custom model training.

| CRISP-ML(Q) phase | Work carried out in GrammarAI |
|---|---|
| **1. Business and Data Understanding** | Defined the grammar-correction objective, intended users, input types, and expected outputs. |
| **2. Data Preparation** | Prepared user input for inference through validation, sentence segmentation, tokenization, and model input-length handling. No custom training dataset was created. |
| **3. Model Engineering** | Selected and integrated the pretrained Vennify T5 grammar-correction checkpoint using PyTorch and Hugging Face Transformers. No project-specific fine-tuning or retraining was performed. |
| **4. Model Evaluation** | Tested representative sentences and paragraphs covering subject-verb agreement, verb tense, articles, prepositions, capitalization, punctuation, and paragraph processing. |
| **5. Deployment** | Built a Streamlit application that displays the original text, corrected output, red-highlighted changes, and suggested corrections. |
| **6. Monitoring and Maintenance** | Identified operational checks and future improvements for consistency, edge cases, latency, and user feedback. Continuous production monitoring is not currently implemented. |

### Quality assurance approach

Quality assurance is considered throughout the lifecycle. The project used representative test inputs to inspect:

- Grammatical corrections such as subject-verb agreement and tense consistency.
- Articles and preposition changes.
- Capitalization and punctuation changes.
- Sentence and paragraph processing.
- Unnecessary rewriting or changes that could affect the original meaning.
- Consistency between the corrected output, red highlights, and suggested-corrections list.

The current evaluation is qualitative and example-based. No benchmark accuracy, BLEU/GLEU score, or other quantitative model-performance claim is reported.

## 📂 Project Structure

```text
Grammar-AI/
│
├── assets/
│   └── grammarai_mascot_quill.png
│
├── src/
│   ├── __init__.py
│   ├── model_utils.py
│   └── correction_manager.py
│
├── app.py
├── GrammarAI_development.ipynb
├── GrammarAI_CRISP_MLQ_Project_Presentation.pptx
├── README.md
├── SETUP.md
├── requirements.txt
└── .gitignore
```

| File / Folder | Description |
|---|---|
| `app.py` | Streamlit application entry point |
| `src/model_utils.py` | Model loading, grammar correction, paragraph processing, text comparison, and highlighting utilities |
| `src/correction_manager.py` | Correction-review utility developed and tested during implementation; not required by the current simplified UI |
| `assets/` | Branding assets used by the application |
| `GrammarAI_development.ipynb` | Development notebook containing model experiments and inference tests |
| `GrammarAI_CRISP_MLQ_Project_Presentation.pptx` | Project presentation explaining the CRISP-ML(Q) lifecycle |
| `requirements.txt` | Python package dependencies |
| `SETUP.md` | Additional setup notes |
| `.gitignore` | Excludes generated and environment-specific files |

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Alley-Py/Grammar-AI.git
cd Grammar-AI
```

### 2. Create and activate a Conda environment

```bash
conda create -n grammarai python=3.10 -y
conda activate grammarai
```

You can also use an existing Python environment, provided it has the required dependencies installed.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch installation fails, install a compatible CPU or CUDA build using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

### 4. Run the application

From the repository root:

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

### 5. Model download

The first model load uses Hugging Face `from_pretrained()` to retrieve the pretrained checkpoint and tokenizer if they are not already cached locally. An internet connection is therefore required for the initial download. Subsequent runs can reuse the local Hugging Face cache.

## 🧩 Implementation Notes

- The current application uses CPU inference, so a dedicated GPU is not required.
- Streamlit resource caching is used to avoid loading the model again on every app interaction.
- The tokenizer applies the input-length configuration defined in the model utility code.
- The current interface displays model-generated corrections directly; it does not include a per-correction accept/reject workflow.
- `correction_manager.py` is retained as a separate project utility, although the simplified deployment does not currently import it.

## ⚠️ Limitations

- Grammar correction quality depends on the pretrained model and the input context.
- The model may occasionally make unnecessary changes or alter the intended meaning.
- Complex, ambiguous, technical, or domain-specific writing may require manual review.
- Red highlights show textual differences, not a guaranteed classification of every grammatical error.
- The current evaluation is based on qualitative examples rather than a formal benchmark.
- Automated production monitoring and a user-feedback collection mechanism have not yet been implemented.

## 🚀 Future Improvements

- Build a larger, structured evaluation set covering diverse grammar-error categories.
- Add quantitative evaluation and human review of correction quality.
- Improve alignment for complex insertions, deletions, and punctuation changes.
- Explore domain-specific evaluation for academic, professional, and technical writing.
- Add latency and failure monitoring.
- Evaluate alternative pretrained grammar-correction models without assuming that fine-tuning is necessary.

## 📚 References and Acknowledgements

- **Pretrained model:** [Vennify — T5 Base Grammar Correction](https://huggingface.co/vennify/t5-base-grammar-correction)
- **Model framework:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- **Deep learning framework:** [PyTorch](https://pytorch.org/)
- **Deployment framework:** [Streamlit](https://streamlit.io/)
- **Sentence segmentation:** [PySBD](https://github.com/nipunsadvilkar/pySBD)
- **Project methodology:** [CRISP-ML(Q): CRoss-Industry Standard Process for the development of Machine Learning applications with Quality assurance methodology](https://arxiv.org/abs/2003.05155)

The pretrained model is credited to its original publisher. Please consult the model card for its licence and usage conditions.

## 👤 Author

**Alley-Py**

[![GitHub](https://img.shields.io/badge/GitHub-Alley--Py-181717?logo=github&logoColor=white)](https://github.com/Alley-Py)

---

<p align="center">

## 📄 License

This project is licensed under the **MIT License**.

Copyright (c) 2026 Alley-Py

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files, to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

---

<p align="center">
  Made with ❤️ by Alley-Py
</p>
  <strong>GrammarAI — Making everyday writing clearer, one correction at a time.</strong>
</p>
