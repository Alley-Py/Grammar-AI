# GrammarAI inference module

## Selected baseline

- Checkpoint: `vennify/t5-base-grammar-correction`
- Framework: PyTorch
- Device: CPU
- Paragraph segmentation: PySBD
- Training/fine-tuning: None

## Files

- `src/model_utils.py`: model loading, sentence correction, paragraph correction, basic structural validation.
- `requirements.txt`: supporting dependencies.

## Use from Jupyter

Run from the project root:

```python
from src.model_utils import load_model, correct_grammar, correct_paragraph, validate_output

tokenizer, model, device = load_model()

sentence = "They was playing cricket when it start raining."
print(correct_grammar(sentence, tokenizer, model))

paragraph = """our application uses machine learning.
the model consider traffic and weather."""
corrected = correct_paragraph(paragraph, tokenizer, model)
print(corrected)
print(validate_output(paragraph, corrected))
```

The first model load downloads the checkpoint from Hugging Face unless it is
already in the local cache.

## Limitations

- The module rejects a sentence that exceeds the token limit rather than silently truncating it.
- PySBD sentence segmentation is heuristic and should be tested on the document types the application will accept.
- `validate_output` checks output structure and length heuristics only. It cannot prove semantic equivalence.
- Vennify is being used as the grammar-correction baseline. Professional rewriting has not been validated for this checkpoint.
