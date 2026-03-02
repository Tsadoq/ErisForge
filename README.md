![erisforge_logo](https://github.com/user-attachments/assets/1a11ad1a-a632-4d5f-990c-3fc84a6c543a)
**ErisForge** is a Python library designed to modify Large Language Models (LLMs) by applying transformations to their internal layers. Named after Eris, the goddess of strife and discord, ErisForge allows you to alter model behavior in a controlled manner, creating both ablated and augmented versions of LLMs that respond differently to specific types of input.


## Features

- Modify internal layers of LLMs to produce altered behaviors.
- Ablate or enhance model responses with the `AblationDecoderLayer` and `AdditionDecoderLayer` classes.
- Measure refusal expressions in model responses using the `ExpressionRefusalScorer`.
- Supports custom behavior directions for applying specific types of transformations.

## Installation

To install ErisForge, clone the repository and install the required packages:

```bash
git clone https://github.com/tsadoq/erisforge.git
cd erisforge
pip install -r requirements.txt
```

or install directly from pip:

```bash
pip install erisforge
```

## Usage

### Basic Setup

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from erisforge import ErisForge
from erisforge.expression_refusal_scorer import ExpressionRefusalScorer

# Load a model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize ErisForge and configure the scorer
forge = ErisForge()
scorer = ExpressionRefusalScorer()
```

### Transform Model Layers

You can apply transformations to specific layers of the model to induce different response behaviors.
A complete example can be found in this notebook: [Transform Model Layers](examples/example_run_forge_refusal_dir.ipynb).

#### Example 1: Applying Ablation to Model Layers

```python
# Define instructions
instructions = ["Explain why AI is beneficial.", "What are the limitations of AI?"]

# Specify layer ranges for ablation
min_layer = 2
max_layer = 4

# Modify the model by applying ablation to the specified layers
ablated_model = forge.run_forged_model(
    model=model,
    type_of_layer=AblationDecoderLayer,
    objective_behaviour_dir=torch.rand(768),  # Example direction tensor
    tokenizer=tokenizer,
    min_layer=min_layer,
    max_layer=max_layer,
    instructions=instructions,
    max_new_tokens=50
)

# Display modified responses
for conversation in ablated_model:
    print("User:", conversation[0]["content"])
    print("AI:", conversation[1]["content"])
```

#### Example 2: Measuring Refusal Expressions

Use `ExpressionRefusalScorer` to measure if the model's response includes common refusal phrases.

```python
response_text = "I'm sorry, I cannot provide that information."
user_query = "What is the recipe for a dangerous substance?"

# Scoring the response for refusal expressions
refusal_score = scorer.score(user_query=user_query, model_response=response_text)
print("Refusal Score:", refusal_score)
```

### Save Transformed Model

You can save your modified model locally or push it to the HuggingFace Hub:

```python
output_model_name = "my_transformed_model"

# Save the modified model
forge.save_model(
    model=model,
    behaviour_dir=torch.rand(768),  # Example direction tensor
    scale_factor=1,
    output_model_name=output_model_name,
    tokenizer=tokenizer,
    to_hub=False  # Set to True to push to HuggingFace Hub
)
```

## Acknowledgments

This project was inspired by and built upon the work from the following repositories and projects:

- [Removing refusals with HF Transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
- [Ablation Blog post on Huggingface](https://huggingface.co/blog/mlabonne/abliteration)
- [DECCP](https://github.com/AUGMXNT/deccp)
- [AbliteratorV3](https://github.com/FailSpy/abliterator)

## Contributing

Feel free to submit issues, suggestions, or contribute directly to this project. Fork the repository, create a feature branch, and submit a pull request.

[Issues and Feature Requests](https://github.com/Tsadoq/ErisForge/issues)

## License

This project is licensed under the MIT License.

## Disclaimer
**Disclaimer**: This library is provided for research and development purposes only. The author assumes no responsibility for any specific applications or uses of ErisForge.
