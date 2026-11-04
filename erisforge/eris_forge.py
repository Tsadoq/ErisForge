import gc
import logging
import platform
import random
import time
from typing import (
    Any,
    Dict,
    List,
    Type,
)

import torch
from torch import (
    Tensor,
)
from tqdm.auto import (
    tqdm,
    trange,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,
)

from erisforge.layers.layers import (
    AblationDecoderLayer,
    AdditionDecoderLayer,
)
from erisforge.scorers.base_scorer import (
    BaseScorer,
)
from erisforge.utils.layer_utils import (
    get_layers_names_by_model,
    identify_model,
)


class Forge:
    def __init__(self, batch_size: int = 10, custom_chat_template: str | None = None):
        """
        Initializes the Forge object.
        :param custom_chat_template: Chat template to use when the tokenizer has no native template.
        """
        self.max_toks = 1
        self.max_iterations: int = 0
        self.batch_size: int = batch_size
        self.custom_chat_template = custom_chat_template
        self.objective_behaviour_instructions: List[str] = []
        self.anti_behaviour_instructions: List[str] = []
        if torch.backends.mps.is_available():
            logging.info("MPS is available.")
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            logging.info("CUDA is available.")
            self.device = torch.device("cuda")
        else:
            logging.info("CPU is available.")
            self.device = torch.device("cpu")

    def load_instructions(
            self,
            objective_behaviour_instructions: List[str],
            anti_behaviour_instructions: List[str],
    ) -> None:
        """
        Loads the instructions for the Forge object.
        :param objective_behaviour_instructions: List of the instructions asking the model to perform the behaviour you want.
        :param anti_behaviour_instructions: List of instructions that ask the model to perform random, usual, tasks.
        :return: None
        """
        logging.info(
            f"Loading instructions, objective_behaviour: {len(objective_behaviour_instructions)}, antiobjective: {len(anti_behaviour_instructions)}"
        )
        self.objective_behaviour_instructions = objective_behaviour_instructions
        self.anti_behaviour_instructions = anti_behaviour_instructions
        self.max_iterations = len(objective_behaviour_instructions) + len(anti_behaviour_instructions)
        logging.info(
            f"Instructions loaded, objective_behaviour: {len(objective_behaviour_instructions)}, antiobjective: {len(anti_behaviour_instructions)}"
        )

    @staticmethod
    def _tokenize(
            tokenizer: PreTrainedTokenizerBase,
            instruction: str,
            bar: tqdm | None = None,
            custom_chat_template: str | None = None,
    ) -> torch.Tensor:
        """
        Tokenizes the instruction.
        :param tokenizer: Tokenizer for a particular model.
        :param instruction: Instruction to be tokenized.
        :param bar: Progress bar object.
        :return: Tokenized instruction in the form of a tensor.
        """
        native_chat_template = getattr(tokenizer, "chat_template", None)
        if native_chat_template is None and custom_chat_template is None:
            logging.error(
                "Warning: your model's tokenizer does not support chat templates. "
                "Falling back to default template."
            )
            tokens: torch.Tensor = tokenizer(
                f"User query: {instruction}\nAI Assistant: ",
                return_tensors="pt",
            )
        else:
            template_kwargs = {}
            if native_chat_template is None:
                template_kwargs["chat_template"] = custom_chat_template
            tokens: torch.Tensor = tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": instruction}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **template_kwargs,
            )
        if bar:
            bar.update(n=1)
        return tokens

    def tokenize_instructions(
            self,
            tokenizer: PreTrainedTokenizerBase | AutoTokenizer | str,
            max_n_objective_behaviour_instruction: int | None = None,
            max_n_antiobjective_instruction: int | None = None,
            disable_tqdm: bool = False,
    ) -> Dict[str, List[Tensor]]:
        """
        Tokenizes the instructions.
        :param tokenizer: Tokenizer for a particular model.
        :param max_n_objective_behaviour_instruction: Maximum number of objective_behaviour instructions to be tokenized.
        :param max_n_antiobjective_instruction: Maximum number of antiobjective instructions to be tokenized.
        :param disable_tqdm: Whether to disable the tqdm progress bar.
        :return: Dictionary containing tokenized objective_behaviour and antiobjective instructions.
        """
        if isinstance(tokenizer, str):
            logging.info(f"Loading tokenizer from {tokenizer}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)

        max_n_objective_behaviour_instruction = (
            min(len(self.objective_behaviour_instructions), max_n_objective_behaviour_instruction)
            if max_n_objective_behaviour_instruction
            else len(self.objective_behaviour_instructions)
        )
        max_n_antiobjective_instruction = (
            min(len(self.anti_behaviour_instructions), max_n_antiobjective_instruction)
            if max_n_antiobjective_instruction
            else len(self.anti_behaviour_instructions)
        )

        objective_behaviour_instructions = random.sample(
            self.objective_behaviour_instructions, max_n_objective_behaviour_instruction
        )
        anti_behaviour_instructions = random.sample(
            self.anti_behaviour_instructions, max_n_antiobjective_instruction
        )

        logging.info(
            f"For tokenization, using {max_n_objective_behaviour_instruction / len(self.objective_behaviour_instructions) * 100:.2f}% objective_behaviour instructions."
        )
        logging.info(
            f"For tokenization, using {max_n_antiobjective_instruction / len(self.anti_behaviour_instructions) * 100:.2f}% antiobjective instructions."
        )

        logging.info("Tokenizing objective_behaviour instructions...")
        with tqdm(total=max_n_objective_behaviour_instruction, desc="Tokenizing objective_behaviour instructions", disable=disable_tqdm) as bar:
            objective_behaviour_instr_tokens = [
                self._tokenize(
                    tokenizer=tokenizer,
                    instruction=instr,
                    bar=bar,
                    custom_chat_template=self.custom_chat_template,
                )
                for instr in objective_behaviour_instructions
            ]

        logging.info("Tokenizing antiobjective instructions...")
        with tqdm(total=max_n_antiobjective_instruction, desc="Tokenizing antiobjective instructions", disable=disable_tqdm) as bar:
            antiobjective_instr_tokens = [
                self._tokenize(
                    tokenizer=tokenizer,
                    instruction=instr,
                    bar=bar,
                    custom_chat_template=self.custom_chat_template,
                )
                for instr in anti_behaviour_instructions
            ]
        logging.info("Tokenization complete.")

        return {
            "objective_behaviour_tokens": objective_behaviour_instr_tokens,
            "antiobjective_tokens": antiobjective_instr_tokens,
        }

    def _generate_new_tokens(
            self,
            model: AutoModelForCausalLM,
            tokens: Tensor,
            bar: tqdm | None = None,
            n_generated_tokens: int = 1,
            streamer: TextStreamer | None = None,
            output_hidden_states: bool = True,
    ) -> GenerateDecoderOnlyOutput:
        """
        Generates new tokens given a prompt.
        :param model: A HuggingFace model.
        :param tokens: Tokenized instruction.
        :param bar: Progress bar object.
        :param n_generated_tokens: Number of tokens to generate.
        :param streamer: TextStreamer object, used for showing the text generation.
        :param output_hidden_states: Whether to output hidden states. Defaults to True.
        :return: Generated tokens.
        """
        if bar:
            bar.update(n=1)

        tokens = tokens.to(self.device)
        params = {
            "use_cache": False,
            "max_new_tokens": n_generated_tokens,
            "return_dict_in_generate": True,
            "output_hidden_states": output_hidden_states,
            **tokens
        }

        if streamer:
            params["streamer"] = streamer

        model.eval()
        with torch.inference_mode():
            output = model.generate(**params)
        return output

    def compute_output(
            self,
            model: AutoModelForCausalLM | str,
            objective_behaviour_tokenized_instructions: List[Tensor],
            anti_behaviour_tokenized_instructions: List[Tensor],
            disable_tqdm: bool = False,
    ) -> Dict[str, List[GenerateDecoderOnlyOutput]]:
        """
        Computes the output for the given instructions.
        :param model: A HuggingFace model.
        :param objective_behaviour_tokenized_instructions: Tokenized objective_behaviour instructions.
        :param anti_behaviour_tokenized_instructions: Tokenized antiobjective instructions.
        :param disable_tqdm: Whether to disable the tqdm progress bar.
        :return: Dictionary containing the outputs for the given instructions.
        """
        if isinstance(model, str):
            logging.info(f"Loading model from {model}")
            model = AutoModelForCausalLM.from_pretrained(
                model,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        else:
            model.to(self.device)
        model.eval()

        logging.info("Generating tokens on objective_behaviour instructions.")
        with tqdm(total=len(objective_behaviour_tokenized_instructions), desc="Generating tokens on objective_behaviour instructions", disable=disable_tqdm) as bar:
            objective_behaviour_outputs = [
                self._generate_new_tokens(
                    model=model,
                    tokens=t,
                    bar=bar,
                    n_generated_tokens=self.max_toks,
                    output_hidden_states=True,
                )
                for t in objective_behaviour_tokenized_instructions
            ]
        logging.info("Completed generating tokens on objective_behaviour instructions.")

        logging.info("Generating tokens on antiobjective instructions.")
        with tqdm(total=len(anti_behaviour_tokenized_instructions), desc="Generating tokens on antiobjective instructions", disable=disable_tqdm) as bar:
            antiobjective_outputs = [
                self._generate_new_tokens(
                    model=model,
                    tokens=t,
                    bar=bar,
                    n_generated_tokens=self.max_toks,
                    output_hidden_states=True,
                )
                for t in anti_behaviour_tokenized_instructions
            ]
        logging.info("Completed generating tokens on antiobjective instructions.")

        return {
            "obj_beh": objective_behaviour_outputs,
            "anti_obj": antiobjective_outputs,
        }

    @staticmethod
    def _print_memory_usage(prefix: str = "") -> None:
        """
        Prints memory usage to better track what is going on under the hood.
        :param prefix: The prefix for the GPU.
        :return: None
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            free = torch.cuda.mem_get_info(0)[0]
            print(f"{prefix}GPU Allocated: {allocated / 1024 ** 3:.2f} GB")
            print(f"{prefix}GPU Cached: {cached / 1024 ** 3:.2f} GB")
            print(f"{prefix}GPU Free: {free / 1024 ** 3:.2f} GB")
            print(f"{prefix}GPU Total: {(allocated + free) / 1024 ** 3:.2f} GB")
        elif platform.system() == "Darwin":
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                used_memory = mem_info.rss / (1024 ** 3)
                available_memory = psutil.virtual_memory().available / (1024 ** 3)
                print(f"{prefix}System Used Memory: {used_memory:.2f} GB")
                print(f"{prefix}System Available Memory: {available_memory:.2f} GB")
                print(f"{prefix}System Total Memory: {(used_memory + available_memory):.2f} GB")
            except ImportError:
                print(f"{prefix}psutil not found. Install with: pip install psutil")
            except Exception as e:
                print(f"{prefix}Error getting system memory info: {e}")
        else:
            print(f"{prefix}Neither CUDA nor macOS detected. Cannot determine memory usage.")

    def _check_memory_usage(self, threshold: float = 0.8) -> None:
        """
        Checks the memory usage and prints a warning if it is above the threshold.
        :param threshold: The threshold for the memory usage.
        :return: None
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.mem_get_info(0)[1]
            used_percentage = allocated / total if total > 0 else 0
            if used_percentage > threshold:
                logging.warning(f"GPU memory usage above {threshold * 100:.0f}%: {used_percentage * 100:.0f}% used.")
        elif platform.system() == "Darwin":
            try:
                import psutil
                process = psutil.Process()
                mem_info = process.memory_info()
                used_memory = mem_info.rss / (1024 ** 3)
                available_memory = psutil.virtual_memory().available / (1024 ** 3)
                if (used_memory / (used_memory + available_memory)) > threshold:
                    logging.warning(
                        f"System memory usage above {threshold * 100:.0f}%: {(used_memory / (used_memory + available_memory) * 100):.0f}% used."
                    )
            except ImportError:
                logging.error("psutil not found. Install with: pip install psutil")
            except Exception as e:
                logging.error(f"Error getting system memory info: {e}")
        else:
            logging.warning("Neither CUDA nor macOS detected. Cannot determine memory usage.")

    def approx_best_objective_behaviour_dir(
            self,
            model: AutoModelForCausalLM | PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase | AutoTokenizer,
            scorer: BaseScorer,
            eval_objective_behaviour_instructions: List[str],
            eval_antiobjective_instructions: List[str],
            min_layer: int | None = None,
            max_layer: int | None = None,
            disable_tqdm: bool = False,
            batch_size: int | None = None,
    ) -> Tensor:
        """
        Finds the approximate best objective_behaviour direction, given the model, tokenizer, scorer, instructions and the layers.
        :param model: A HuggingFace model.
        :param tokenizer: Tokenizer for a particular model.
        :param scorer: Scorer object.
        :param eval_objective_behaviour_instructions: Instructions for evaluating the objective_behaviour.
        :param eval_antiobjective_instructions: Instructions for evaluating the antiobjective.
        :param min_layer: First layer to be considered for computing the best direction.
        :param max_layer: Last layer to be considered for computing the best direction.
        :param disable_tqdm: Whether to disable the tqdm progress bar.
        :param batch_size: Batch size for the model.
        :return: The approximate best objective_behaviour direction.
        """
        if min_layer is None:
            min_layer = max(int(len(model.model.layers) * 0.2), 1)
        if max_layer is None:
            max_layer = min(int(len(model.model.layers) * 0.9), len(model.model.layers) - 2)

        batch_size = batch_size if batch_size else self.batch_size

        model.to(self.device)
        model.eval()

        logging.info(f"Using layers from {min_layer} to {max_layer} for computing best direction.")
        logging.info(f"\n============== Scores will be computed for {max_layer - min_layer} different layers ==============")

        score_x_layer = []
        logging.info("Tokenizing evaluation instructions...")
        with tqdm(total=len(eval_objective_behaviour_instructions), desc="Tokenizing objective_behaviour Eval Instructions set", disable=disable_tqdm) as bar:
            obj_beh_toks = [
                self._tokenize(
                    tokenizer=tokenizer,
                    instruction=instr,
                    bar=bar,
                    custom_chat_template=self.custom_chat_template,
                )
                for instr in eval_objective_behaviour_instructions
            ]
        with tqdm(total=len(eval_antiobjective_instructions), desc="Tokenizing antiobjective Eval Instructions set", disable=disable_tqdm) as bar:
            anti_obj_toks = [
                self._tokenize(
                    tokenizer=tokenizer,
                    instruction=instr,
                    bar=bar,
                    custom_chat_template=self.custom_chat_template,
                )
                for instr in eval_antiobjective_instructions
            ]

        logging.info("Computing output for evaluation instructions...")
        d_out = self.compute_output(
            model=model,
            objective_behaviour_tokenized_instructions=obj_beh_toks,
            anti_behaviour_tokenized_instructions=anti_obj_toks,
            disable_tqdm=disable_tqdm,
        )
        self._check_memory_usage()

        logging.info("Freeing memory from tokenized instructions...")
        self.free_memory([obj_beh_toks, anti_obj_toks])
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("Freed Memory")

        for layer_idx in trange(min_layer, max_layer, desc="Finding best direction", disable=disable_tqdm, colour="green"):
            start_time = time.time()
            logging.info(f"Computing objective_behaviour direction for layer: {layer_idx}")
            tmp_obj_beh_dir = self.compute_objective_behaviour_direction(
                model=model,
                objective_behaviour_outputs=d_out["obj_beh"],
                antiobjective_outputs=d_out["anti_obj"],
                layer=layer_idx,
            ).cpu()
            self._check_memory_usage()

            logging.info("Running inference on objective_behaviour instructions (Ablated Model)...")
            conversations_ablated = []
            conversations_added = []

            for batch_start in trange(0, len(eval_objective_behaviour_instructions), batch_size, desc="Ablation on objective_behaviour", disable=disable_tqdm, colour="blue"):
                batch_instructions = eval_objective_behaviour_instructions[batch_start:batch_start + batch_size]
                conversations_ablated.extend(
                    self.run_forged_model(
                        model=model,
                        type_of_layer=AblationDecoderLayer,
                        objective_behaviour_dir=tmp_obj_beh_dir,
                        tokenizer=tokenizer,
                        min_layer=min_layer,
                        max_layer=max_layer,
                        instructions=batch_instructions,
                        max_new_tokens=100,
                        stream=False,
                        disable_tqdm=True,
                    )
                )
                gc.collect()
                torch.cuda.empty_cache()

            logging.info("Running inference on anti_objective_behaviour instructions (Added Model)...")
            for batch_start in trange(0, len(eval_antiobjective_instructions), batch_size, desc="Addition on antiobjective", disable=disable_tqdm, colour="blue"):
                batch_instructions = eval_antiobjective_instructions[batch_start:batch_start + batch_size]
                conversations_added.extend(
                    self.run_forged_model(
                        model=model,
                        type_of_layer=AdditionDecoderLayer,
                        objective_behaviour_dir=tmp_obj_beh_dir,
                        tokenizer=tokenizer,
                        min_layer=layer_idx,
                        max_layer=layer_idx + 1,
                        instructions=batch_instructions,
                        max_new_tokens=100,
                        stream=False,
                        disable_tqdm=True,
                    )
                )
                gc.collect()
                torch.cuda.empty_cache()

            objective_behaviour_score = sum([
                scorer.score(
                    model_response=conv[-1]["content"],
                    user_query=conv[-2]["content"]
                )
                for conv in conversations_ablated
            ])
            antiobjective_score = 1 - sum([
                scorer.score(
                    model_response=conv[-1]["content"],
                    user_query=conv[-2]["content"]
                )
                for conv in conversations_added
            ])

            score_x_layer.append({
                "layer": layer_idx,
                "score": (objective_behaviour_score - antiobjective_score) / 2,
                "dir": tmp_obj_beh_dir,
            })
            self._check_memory_usage()

            del tmp_obj_beh_dir, conversations_ablated, conversations_added
            gc.collect()
            torch.cuda.empty_cache()

            end_time = time.time()
            logging.info(
                f"\nLayer {layer_idx} done in {end_time - start_time:.2f} seconds. Objective_behaviour score: {objective_behaviour_score:.2f}, Antiobjective score: {antiobjective_score:.2f}"
            )

        score_x_layer = sorted(score_x_layer, key=lambda x: x["score"], reverse=True)
        return score_x_layer[0]["dir"]

    @staticmethod
    def _replace_layers(
            new_layer: Type[torch.nn.Module],
            max_layer: int,
            min_layer: int,
            model: AutoModelForCausalLM | PreTrainedModel,
            direction: Tensor,
            disable_tqdm: bool = False,
    ) -> AutoModelForCausalLM | PreTrainedModel:
        """
        Replaces the layers of the model.
        :param new_layer: Type of layer to be replaced.
        :param max_layer: Maximum layer to be replaced.
        :param min_layer: Minimum layer to be replaced.
        :param model: A HuggingFace model.
        :param direction: Direction tensor.
        :param disable_tqdm: Whether to disable the tqdm progress bar.
        :return: Model with replaced layers.
        """
        for layer_idx in trange(min_layer, max_layer, desc="Replacing model layers", disable=disable_tqdm):
            current_layer = model.model.layers[layer_idx]
            if isinstance(current_layer, (AblationDecoderLayer, AdditionDecoderLayer)):
                base_layer = current_layer.original_layer
            else:
                base_layer = current_layer
            replaced_layer = new_layer(original_layer=base_layer, direction=direction)
            for attr_name in dir(base_layer):
                if attr_name.startswith("__"):
                    continue
                if not hasattr(replaced_layer, attr_name):
                    try:
                        setattr(replaced_layer, attr_name, getattr(base_layer, attr_name))
                    except Exception:
                        logging.error(f"Error setting attribute {attr_name} on layer {layer_idx}")
                        pass

            model.model.layers[layer_idx] = replaced_layer
        return model
    def compute_objective_behaviour_direction(
            self,
            model: AutoModelForCausalLM | PreTrainedModel,
            objective_behaviour_outputs: List[GenerateDecoderOnlyOutput],
            antiobjective_outputs: List[GenerateDecoderOnlyOutput],
            layer: int | None = None,
    ) -> Tensor:
        """
        Computes the objective_behaviour direction given a layer.
        :param model: A HuggingFace model.
        :param objective_behaviour_outputs: Objective_behaviour outputs.
        :param antiobjective_outputs: Antiobjective outputs.
        :param layer: Layer to be considered for computing the objective_behaviour direction.
        :return: Objective_behaviour direction.
        """
        if layer is None:
            layer = int(len(model.model.layers) * 0.6)

        with torch.inference_mode():
            objective_behaviour_mean = torch.stack(
                [
                    output.hidden_states[0][layer][:, -self.max_toks:, :].mean(dim=1)
                    for output in objective_behaviour_outputs
                ]
            ).mean(dim=0)
            antiobjective_mean = torch.stack(
                [
                    output.hidden_states[0][layer][:, -self.max_toks:, :].mean(dim=1)
                    for output in antiobjective_outputs
                ]
            ).mean(dim=0)

        objective_behaviour_dir = objective_behaviour_mean - antiobjective_mean
        objective_behaviour_dir = (
                objective_behaviour_dir / objective_behaviour_dir.norm()
        )

        return objective_behaviour_dir

    def run_forged_model(
            self,
            model: AutoModelForCausalLM | PreTrainedModel,
            objective_behaviour_dir: Tensor,
            tokenizer: PreTrainedTokenizerBase | AutoTokenizer,
            type_of_layer: Type[torch.nn.Module] | None = None,
            min_layer: int | None = None,
            max_layer: int | None = None,
            instructions: List[str] | None = None,
            tokenized_instructions: List[Tensor] | None = None,
            max_new_tokens: int = 100,
            stream: bool = False,
            disable_tqdm: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """
        Runs the forged model.
        :param model: A HuggingFace model.
        :param objective_behaviour_dir: Objective_behaviour direction.
        :param tokenizer: Tokenizer for a particular model.
        :param type_of_layer: Type of layer to be replaced.
        :param min_layer: Minimum layer to be replaced.
        :param max_layer: Maximum layer to be replaced.
        :param instructions: Instructions to be used for the forged model.
        :param tokenized_instructions: Tokenized instructions to be used for the forged model.
        :param max_new_tokens: Maximum number of tokens to be generated.
        :param stream: Whether to show as text the generation.
        :param disable_tqdm: Whether to disable the tqdm progress bar.
        :return: List of conversations.
        """
        if min_layer is None:
            min_layer = max(int(len(model.model.layers) * 0.2), 1)
        if max_layer is None:
            max_layer = min(int(len(model.model.layers) * 0.8), len(model.model.layers) - 2)

        new_model = self._replace_layers(
            new_layer=(type_of_layer if type_of_layer else AblationDecoderLayer),
            max_layer=max_layer,
            min_layer=min_layer,
            model=model,
            direction=objective_behaviour_dir,
            disable_tqdm=disable_tqdm,
        )
        new_model.eval()

        if tokenized_instructions:
            logging.info("Using provided tokenized instructions. No need to tokenize again.")
            instr_tokens = tokenized_instructions
        elif instructions:
            logging.info("Tokenizing instructions for newly forged model.")
            with tqdm(total=len(instructions), desc="Tokenizing instructions for newly forged model", disable=disable_tqdm) as bar:
                instr_tokens = [
                    self._tokenize(
                        tokenizer=tokenizer,
                        instruction=instr,
                        bar=bar,
                        custom_chat_template=self.custom_chat_template,
                    )
                    for instr in instructions
                ]
        else:
            raise ValueError("Either instructions or tokenized instructions must be provided.")

        logging.info("Generating tokens for newly forged model.")
        with tqdm(total=len(instr_tokens), desc="Generating tokens for newly forged model", disable=disable_tqdm) as bar:
            with torch.inference_mode():
                encoded_responses = [
                    self._generate_new_tokens(
                        model=new_model,
                        tokens=t,
                        bar=bar,
                        n_generated_tokens=max_new_tokens,
                        streamer=TextStreamer(tokenizer) if stream else None,
                        output_hidden_states=True,
                    )
                    for t in instr_tokens
                ]

        self.free_memory([new_model], silent=not disable_tqdm)
        gc.collect()
        torch.cuda.empty_cache()

        conversations = []
        for enc_resp, instr in zip(encoded_responses, instructions or []):
            conversations.append([
                {"role": "user", "content": instr},
                {"role": "assistant", "content": tokenizer.decode(enc_resp.sequences[0].tolist(), skip_special_tokens=True)},
            ])

        return conversations

    def evaluate_base_model(
            self,
            model: AutoModelForCausalLM | PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase | AutoTokenizer,
            instructions: List[str] | None = None,
            tokenized_instructions: List[Tensor] | None = None,
            max_new_tokens: int = 100,
            stream: bool = False,
            disable_tqdm: bool = False,
    ) -> List[List[Dict[str, Any]]]:
        """
        Runs the forged model.
        :param model: A HuggingFace model.
        :param tokenizer: Tokenizer for a particular model.
        :param instructions: Instructions to be used for the forged model.
        :param tokenized_instructions: Tokenized instructions to be used for the forged model.
        :param max_new_tokens: Maximum number of tokens to be generated.
        :param stream: Whether to show as text the generation.
        :param disable_tqdm: Whether to disable the tqdm progress bar.
        :return: List of conversations.
        """
        model.to(self.device)
        model.eval()

        if tokenized_instructions:
            logging.info("Using provided tokenized instructions. No need to tokenize again.")
            instr_tokens = tokenized_instructions
        elif instructions:
            with tqdm(total=len(instructions), desc="Tokenizing instructions...", disable=disable_tqdm) as bar:
                instr_tokens = [
                    self._tokenize(
                        tokenizer=tokenizer,
                        instruction=instr,
                        bar=bar,
                        custom_chat_template=self.custom_chat_template,
                    )
                    for instr in instructions
                ]
        else:
            raise ValueError("Either instructions or tokenized instructions must be provided.")

        logging.info("Running inference on base model...")
        with tqdm(total=len(instr_tokens), desc="Inference on base model...", disable=disable_tqdm) as bar:
            encoded_responses = [
                self._generate_new_tokens(
                    model=model,
                    tokens=t,
                    bar=bar,
                    n_generated_tokens=max_new_tokens,
                    streamer=TextStreamer(tokenizer) if stream else None,
                    output_hidden_states=False,
                )
                for t in instr_tokens
            ]

        conversations = []
        for enc_resp, instr in zip(encoded_responses, instructions or []):
            conversations.append([
                {"role": "user", "content": instr},
                {"role": "assistant", "content": tokenizer.decode(enc_resp.sequences[0].tolist(), skip_special_tokens=True)},
            ])

        self.free_memory([instructions, encoded_responses])
        gc.collect()
        torch.cuda.empty_cache()

        return conversations

    def _modify_tensor(
            self,
            tensor: Tensor,
            behaviour_dir: Tensor,
            scale_factor: float = 1.0,
    ) -> Tensor:
        """
        Modifies the tensor applying the behaviour direction.
        :param tensor: Tensor to be modified.
        :param behaviour_dir: Behaviour direction.
        :param scale_factor: Scale factor, must be between -1.0 and 1.0. If negative, induces the behaviour.
        :return: Modified tensor.
        """
        if abs(scale_factor) > 1.0:
            raise ValueError("The scale factor must be between -1.0 and 1.0.")

        tensor_float32 = tensor.to(torch.float32).to(self.device)
        behaviour_dir_float32 = behaviour_dir.to(torch.float32).to(self.device)

        if behaviour_dir_float32.dim() > 1:
            behaviour_dir_float32 = behaviour_dir_float32.view(-1)

        outer_dir = torch.outer(behaviour_dir_float32, behaviour_dir_float32)
        tensor_float32 -= scale_factor * torch.matmul(outer_dir, tensor_float32)
        tensor_modified = tensor_float32.to(torch.bfloat16)
        return torch.nn.Parameter(tensor_modified)

    def save_model(
            self,
            model: AutoModelForCausalLM | PreTrainedModel,
            behaviour_dir: Tensor,
            scale_factor: float = 1.0,
            min_layer: int | None = None,
            max_layer: int | None = None,
            output_model_name: str = None,
            tokenizer: PreTrainedTokenizerBase | AutoTokenizer = None,
            to_hub: bool = False,
            model_architecture: str = "gemma",
            disable_tqdm: bool = False,
    ) -> AutoModelForCausalLM | PreTrainedModel:
        """
        Modifies the layers and saves the model (to disk or to the HuggingFace Hub).
        :param model: A HuggingFace model.
        :param behaviour_dir: Behaviour direction.
        :param scale_factor: Scale factor, must be between -1.0 and 1.0. If negative, induces the behaviour.
        :param min_layer: Minimum layer to be modified.
        :param max_layer: Maximum layer to be modified.
        :param output_model_name: Name of the (new) model, useful if pushed to hub or saved somewhere.
        :param tokenizer: Tokenizer for a particular model, useful if pushed to hub or saved somewhere.
        :param to_hub: Whether to push the model to the HuggingFace Hub.
        :param model_architecture: Model architecture, needed to identify what are the layers to be modified. If not specified, will try to find the layers to be modified by trying all possibilities.
        :param disable_tqdm: Whether to disable the tqdm progress bar.
        :return: Modified model.
        """
        if abs(scale_factor) > 1.0:
            raise ValueError("The scale factor must be between -1.0 and 1.0.")
        if not model_architecture:
            logging.warning("No model architecture provided. Trying to identify the model architecture automatically.")
            model_architecture = identify_model(model.model)

        layer_names = get_layers_names_by_model(model_architecture.lower())
        custom_model = model.model

        if min_layer is None:
            min_layer = max(int(len(model.model.layers) * 0.2), 2)
        if max_layer is None:
            max_layer = min(int(len(model.model.layers) * 0.8), len(model.model.layers) - 3)

        model.eval()
        for layer_idx in trange(min_layer, max_layer, desc="Modifying model layers", disable=disable_tqdm):
            layer = custom_model.layers[layer_idx]
            for attr_path in layer_names.values():
                parts = attr_path.split(".")
                target = layer
                for part in parts[:-1]:
                    target = getattr(target, part)
                weight_attr = getattr(target, parts[-1])
                modified_weight = self._modify_tensor(
                    tensor=weight_attr,
                    behaviour_dir=behaviour_dir,
                    scale_factor=scale_factor,
                )
                setattr(target, parts[-1], modified_weight)
        if output_model_name:
            model.save_pretrained(output_model_name, push_to_hub=to_hub)
            if tokenizer:
                tokenizer.save_pretrained(output_model_name, push_to_hub=to_hub)
        else:
            logging.warning("No output model name provided. Model not saved to disk nor pushed to hub.")
        return model

    def free_memory(self, list_of_variables: List[Any], silent: bool = False) -> None:
        """
        Frees the memory.
        :param list_of_variables: List of variables to be deleted.
        :param silent: Whether to print the warning message.
        :return: None
        """
        logging.info(f"Freeing memory for {len(list_of_variables)} variables.") if not silent else None
        for var in list_of_variables:
            try:
                del var
            except Exception as e:
                logging.error(f"Error deleting variable: {e}")
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
