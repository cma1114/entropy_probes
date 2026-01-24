"""
Create difficulty-filtered dataset by selecting questions the model gets right/wrong.

Creates a .jsonl file that plugs directly into the existing pipeline:
  data/{dataset}_difficulty_filtered.jsonl

Then use with existing scripts:
  identify_mc_correlate.py:  DATASET = "TriviaMC_difficulty_filtered"
  test_cross_dataset_transfer.py: auto-discovers it

Requires GPU to run the model and determine correct/incorrect.
"""

from pathlib import Path
import json
import random
import sys

import numpy as np
from tqdm import tqdm

from core import (
    load_model_and_tokenizer,
    should_use_chat_template,
    load_questions,
)
from tasks import format_direct_prompt


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DATASET = "TriviaMC"
N_CORRECT = 250
N_INCORRECT = 250
SEED = 42
BATCH_SIZE = 8

# Quantization (for large models)
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

# Estimation parameters
ESTIMATED_ACCURACY = 0.87
SAFETY_MARGIN = 1.3

DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# MAIN
# =============================================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    DATA_DIR.mkdir(exist_ok=True)

    output_path = DATA_DIR / f"{DATASET}_difficulty_filtered.jsonl"

    print("=" * 70)
    print("DIFFICULTY-FILTERED DATASET CREATION")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Source: {DATASET}")
    print(f"Target: {N_CORRECT} correct + {N_INCORRECT} incorrect")
    print(f"Output: {output_path}")
    print()

    # --- Calculate how many questions to evaluate ---
    required = int(np.ceil(N_INCORRECT / (1 - ESTIMATED_ACCURACY) * SAFETY_MARGIN))
    print(f"Step 1: Need ~{required} questions to get {N_INCORRECT} incorrect at {ESTIMATED_ACCURACY:.0%} accuracy")

    # --- Load model ---
    print(f"\nStep 2: Loading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        MODEL,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    use_chat_template = should_use_chat_template(MODEL, tokenizer)

    # --- Load questions ---
    print(f"\nStep 3: Loading {required} questions from {DATASET}...")
    questions = load_questions(DATASET, num_questions=required, seed=SEED)
    print(f"  Loaded {len(questions)} questions")

    option_keys = list(questions[0]["options"].keys())
    option_token_ids = [tokenizer.encode(k, add_special_tokens=False)[0] for k in option_keys]

    # --- Run model to determine correct/incorrect ---
    print(f"\nStep 4: Running model to classify correct/incorrect...")

    correct_questions = []
    incorrect_questions = []

    for batch_start in tqdm(range(0, len(questions), BATCH_SIZE)):
        batch = questions[batch_start:batch_start + BATCH_SIZE]

        prompts = [format_direct_prompt(q, tokenizer, use_chat_template)[0] for q in batch]

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get prediction for each item in batch
        for i, q in enumerate(batch):
            # Get logits from last position (works with left-padding)
            last_logits = logits[i, -1, :]

            # Get logits for option tokens
            option_logits = last_logits[option_token_ids].cpu().numpy()
            predicted_idx = np.argmax(option_logits)
            predicted = option_keys[predicted_idx]

            is_correct = predicted == q["correct_answer"]

            # Store question with EXACT option ordering (not text-based)
            # This preserves the A/B/C/D positions so the model sees identical prompts
            q_original = {
                "qid": q.get("qid", q.get("id", f"{DATASET}_{batch_start + i}")),
                "question": q["question"],
                "options": q["options"],  # Exact A/B/C/D dict
                "correct_answer": q["correct_answer"],  # Letter, not text
            }

            if is_correct:
                correct_questions.append(q_original)
            else:
                incorrect_questions.append(q_original)

    print(f"\n  Correct: {len(correct_questions)}")
    print(f"  Incorrect: {len(incorrect_questions)}")
    actual_acc = len(correct_questions) / len(questions)
    print(f"  Accuracy: {actual_acc:.1%}")

    # --- Check if we have enough ---
    if len(correct_questions) < N_CORRECT:
        print(f"\nERROR: Not enough correct ({len(correct_questions)} < {N_CORRECT})")
        sys.exit(1)

    if len(incorrect_questions) < N_INCORRECT:
        print(f"\nERROR: Not enough incorrect ({len(incorrect_questions)} < {N_INCORRECT})")
        print(f"  Model accuracy too high. Options:")
        print(f"    - Reduce N_INCORRECT to {len(incorrect_questions)}")
        print(f"    - Use harder dataset")
        sys.exit(1)

    # --- Sample and write output ---
    print(f"\nStep 5: Sampling {N_CORRECT} correct + {N_INCORRECT} incorrect...")

    selected_correct = random.sample(correct_questions, N_CORRECT)
    selected_incorrect = random.sample(incorrect_questions, N_INCORRECT)
    selected = selected_correct + selected_incorrect
    random.shuffle(selected)

    print(f"\nStep 6: Writing {output_path}...")
    with open(output_path, "w") as f:
        for q in selected:
            f.write(json.dumps(q) + "\n")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Created: {output_path}")
    print(f"  {len(selected)} questions ({N_CORRECT} correct + {N_INCORRECT} incorrect)")
    print()
    print("Now run the normal pipeline:")
    print(f'  1. Edit identify_mc_correlate.py: DATASET = "{DATASET}_difficulty_filtered"')
    print(f"  2. python identify_mc_correlate.py")
    print(f"  3. python test_cross_dataset_transfer.py")


if __name__ == "__main__":
    import torch
    main()
