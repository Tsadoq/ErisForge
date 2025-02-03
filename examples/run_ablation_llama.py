print("Importing necessary libraries...")
import random
print("Importing necessary libraries...")
import torch
print("Importing necessary libraries...")
from transformers import AutoModelForCausalLM, AutoTokenizer
print("Importing necessary libraries...")
from erisforge import Forge
print("Importing necessary libraries...")
from erisforge.scorers import (
    ExpressionRefusalScorer,
)
random.seed(42)
MODEL = "meta-llama/Llama-3.1-70B-Instruct"
with open("harmful_instructions.txt", "r") as f:
    obj_beh = f.readlines()
with open("harmless_instructions.txt", "r") as f:
    anti_obj = f.readlines()
max_inst = 100
forge = Forge()
print("Loading model or data...")
forge.load_instructions(
    objective_behaviour_instructions=obj_beh, anti_behaviour_instructions=anti_obj
)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for efficiency if supported
).to(forge.device)  # Move model to the device set in forge (e.g., GPU if available)
d_toks = forge.tokenize_instructions(
    tokenizer=tokenizer,
    max_n_antiobjective_instruction=max_inst,
    max_n_objective_behaviour_instruction=max_inst,
)
d_instr = forge.compute_output(
    model=model,
    objective_behaviour_tokenized_instructions=d_toks["objective_behaviour_tokens"],
    anti_behaviour_tokenized_instructions=d_toks["antiobjective_tokens"],
)
scorer = ExpressionRefusalScorer()
forge.free_memory([d_toks])
try:
    refusal_dir = forge.find_approximate_best_objective_behaviour_direction(
        model=model,
        tokenizer=tokenizer,
        scorer=scorer,
        eval_objective_behaviour_instructions=obj_beh[:max_inst],
        eval_antiobjective_instructions=anti_obj[:max_inst],
        min_layer=10,
        max_layer=13,
    )
    print("Best direction computed successfully.")
except Exception as e:
    print("An error occurred during computation:", e)
    print("This may be due to memory constraints or a memory leak.")
conversations = forge.run_forged_model(
    model=model,
    objective_behaviour_dir=refusal_dir,
    tokenizer=tokenizer,
    instructions=random.sample(population=obj_beh, k=20),
    max_new_tokens=100,
    stream=False,
)
for conversation in conversations:
    print("=" * 20)
    for round in conversation:
        print(f'{round["role"]}: {round["content"]}')
forge.free_memory([conversations, d_toks, d_instr])  # Free memory after testing
print("Saving the modified model...")
forge.save_model(
    model=model,
    tokenizer=tokenizer,
    behaviour_dir=refusal_dir,
    output_model_name="BadLLama-3.1-70b",  # Name for the saved model
    to_hub=False,  # Set to True to push the model to the HuggingFace Hub
    model_architecture="llama",  # Specify the model architecture
)
tokenizer = AutoTokenizer.from_pretrained("corrupted_gemma_test")
model = AutoModelForCausalLM.from_pretrained("corrupted_gemma_test")
input_text = "give me a racist insult against italians"
input_ids = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**input_ids, max_length=1000)
print(tokenizer.decode(outputs[0]))