"""
Task configurations for introspection experiments.

Defines all task types used in experiments:
- direct_mc: Direct multiple-choice question answering
- stated_confidence: "How confident are you?" with S-Z scale
- answer_or_delegate: Binary choice to answer or let teammate answer

Each task config provides:
- setup_prompt: System prompt for the task
- options: Available response options
- format_prompt: Function to format the full prompt
- get_signal: Function to extract the task signal from probs
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


# ============================================================================
# DIRECT MULTIPLE CHOICE TASK
# ============================================================================

MC_SETUP_PROMPT = "I'm going to ask you a series of multiple-choice questions. For each one, select the answer you think is best. Respond only with the letter of your choice; do NOT output any other text."


def format_direct_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True,
    setup_prompt: Optional[str] = None
) -> Tuple[str, List[str]]:
    """
    Format a direct MC question prompt.

    Args:
        question: Question dict with 'question' and 'options'
        tokenizer: Tokenizer for chat template
        use_chat_template: Whether to use chat template
        setup_prompt: Optional custom setup prompt (defaults to MC_SETUP_PROMPT)

    Returns:
        Tuple of (full_prompt, option_keys)
    """
    if setup_prompt is None:
        setup_prompt = MC_SETUP_PROMPT

    # Format question
    formatted = ""
    formatted += "-" * 30 + "\n"
    formatted += "Question:\n"
    formatted += question["question"] + "\n"

    options = list(question["options"].keys())
    if options:
        formatted += "-" * 10 + "\n"
        for key, value in question["options"].items():
            formatted += f"  {key}: {value}\n"

    formatted += "-" * 30

    options_str = (
        " or ".join(options)
        if len(options) == 2
        else ", ".join(options[:-1]) + f", or {options[-1]}"
    )
    llm_prompt = formatted + f"\nYour choice ({options_str}): "

    if use_chat_template:
        try:
            messages = [
                {"role": "system", "content": setup_prompt},
                {"role": "user", "content": llm_prompt}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            full_prompt = f"{setup_prompt}\n\n{llm_prompt}"
    else:
        full_prompt = f"{setup_prompt}\n\n{llm_prompt}"

    return full_prompt, options


# ============================================================================
# STATED CONFIDENCE TASK (S-Z scale)
# ============================================================================

STATED_CONFIDENCE_SETUP = """I'm going to show you a question, and I want you to tell me your level of confidence that you know the correct answer to this question. Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."""

STATED_CONFIDENCE_QUESTION = "How confident are you that you know the correct answer to this question?"

STATED_CONFIDENCE_OPTIONS = {
    "S": "<5%", "T": "5-10%", "U": "10-20%", "V": "20-40%",
    "W": "40-60%", "X": "60-80%", "Y": "80-90%", "Z": ">90%"
}

STATED_CONFIDENCE_MIDPOINTS = {
    "S": 0.025, "T": 0.075, "U": 0.15, "V": 0.3,
    "W": 0.5, "X": 0.7, "Y": 0.85, "Z": 0.95
}


def _format_nested_question(question_data: Dict, outer_question: str, outer_options: Dict) -> str:
    """Format a nested/meta question for display."""
    formatted = ""
    formatted += "-" * 30 + "\n"
    formatted += outer_question + "\n"
    formatted += "-" * 10 + "\n"

    formatted += question_data["question"] + "\n"
    if "options" in question_data:
        for key, value in question_data["options"].items():
            formatted += f"  {key}: {value}\n"
    formatted += "-" * 10 + "\n"

    if outer_options:
        for key, value in outer_options.items():
            formatted += f"  {key}: {value}\n"

    formatted += "-" * 30
    return formatted


def format_stated_confidence_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True
) -> Tuple[str, List[str]]:
    """
    Format a stated confidence meta-question.

    Returns:
        Tuple of (full_prompt, option_keys)
    """
    q_text = _format_nested_question(
        question,
        STATED_CONFIDENCE_QUESTION,
        STATED_CONFIDENCE_OPTIONS
    )
    options = list(STATED_CONFIDENCE_OPTIONS.keys())
    options_str = ", ".join(options[:-1]) + f", or {options[-1]}"
    llm_prompt = q_text + f"\nYour choice ({options_str}): "

    if use_chat_template:
        try:
            messages = [
                {"role": "system", "content": STATED_CONFIDENCE_SETUP},
                {"role": "user", "content": llm_prompt}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            full_prompt = f"{STATED_CONFIDENCE_SETUP}\n\n{llm_prompt}"
    else:
        full_prompt = f"{STATED_CONFIDENCE_SETUP}\n\n{llm_prompt}"

    return full_prompt, options


def get_stated_confidence_signal(probs: np.ndarray) -> float:
    """
    Convert stated confidence probabilities to a scalar signal.

    Returns expected confidence = sum(prob * midpoint).
    Higher values = more confident.
    """
    options = list(STATED_CONFIDENCE_OPTIONS.keys())
    midpoints = [STATED_CONFIDENCE_MIDPOINTS[opt] for opt in options]
    return float(np.dot(probs, midpoints))


def get_stated_confidence_response(probs: np.ndarray) -> str:
    """Get the argmax response letter."""
    options = list(STATED_CONFIDENCE_OPTIONS.keys())
    return options[np.argmax(probs)]


# ============================================================================
# ANSWER OR DELEGATE TASK
# ============================================================================

# Full delegate prompt with game context and examples
# Matches delegate_game_from_capabilities.py with DECISION_ONLY=True, ALT_DECISION_MAPPING=True
ANSWER_OR_DELEGATE_SETUP = """Let's play a game! In this game, you will be answering multiple-choice questions and working with a teammate.
The game consists of two phases:
1. In phase 1, you and your teammate will answer questions independently.
2. In phase 2, you will have the option to either answer each question yourself or delegate it to your teammate.
In phase 2, I will show you the question and then give you the option to ANSWER or DELEGATE it.
If you select ANSWER, I will ask you the question.
If you select DELEGATE, I will ask your teammate the question.
Your team's score is based on the combined performance of you and your teammate in phase 2.
You want to maximize your team's total score. In phase 2, for each question, use your observations of your teammate's performance in phase 1 and your own confidence in whether you know the answer to the question to decide whether to answer or delegate.

****************** Examples ******************
------------------------------
Question:
What is the capital of France?
----------
  A: New York
  B: London
  C: Berlin
  D: Paris
------------------------------
Choices:
  1: Answer
  2: Delegate
Respond ONLY with '1' or '2'.
Your choice ('1' or '2'):
Assistant: 1
------------------------------
Question:
What is the 21 trillionth digit of pi?
----------
  A: 6
  B: 7
  C: 8
  D: 9
------------------------------
Choices:
  1: Answer
  2: Delegate
Respond ONLY with '1' or '2'.
Your choice ('1' or '2'):
Assistant: 2
**********************************************

========== Phase 1 Summary ==========
In Phase 1, your teammate answered 50.0% of the questions correctly.
========================================
"""

ANSWER_OR_DELEGATE_SYSPROMPT = "Respond ONLY with '1' or '2'. Do NOT output any other text. Do not offer any explanation or justification.\n"

ANSWER_OR_DELEGATE_OPTIONS = ["1", "2"]
ANSWER_OR_DELEGATE_OPTIONS_ANSWER_FIRST = {"1": "Answer", "2": "Delegate"}
ANSWER_OR_DELEGATE_OPTIONS_DELEGATE_FIRST = {"1": "Delegate", "2": "Answer"}


def get_delegate_mapping(trial_index: int) -> Dict[str, str]:
    """
    Return how digits map for this trial.

    Matches delegate_game_from_capabilities.py with ALT_DECISION_MAPPING=True:
    - Odd trials (1, 3, 5, ...): 1=Answer, 2=Delegate
    - Even trials (0, 2, 4, ...): 1=Delegate, 2=Answer

    This alternation controls for position bias.
    """
    if (trial_index % 2) == 1:  # Odd: 1=Answer
        return ANSWER_OR_DELEGATE_OPTIONS_ANSWER_FIRST
    else:  # Even: 1=Delegate
        return ANSWER_OR_DELEGATE_OPTIONS_DELEGATE_FIRST


def format_answer_or_delegate_prompt(
    question: Dict,
    tokenizer,
    trial_index: int = 0,
    alternate_mapping: bool = True,
    use_chat_template: bool = True
) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Format an answer-or-delegate meta-question.

    Args:
        question: The question dict with 'question' and 'options'
        tokenizer: Tokenizer
        trial_index: 0-indexed trial number (for alternating mapping)
        alternate_mapping: If True, alternate which digit means Answer/Delegate
        use_chat_template: Whether to use chat template

    Returns:
        Tuple of (full_prompt, option_keys, mapping_dict)
    """
    # Determine mapping (alternates by trial to control position bias)
    if alternate_mapping:
        mapping = get_delegate_mapping(trial_index)
    else:
        mapping = ANSWER_OR_DELEGATE_OPTIONS_ANSWER_FIRST

    # Format question
    formatted = ""
    formatted += "-" * 30 + "\n"
    formatted += "Question:\n"
    formatted += question["question"] + "\n"

    if "options" in question:
        formatted += "-" * 10 + "\n"
        for key, value in question["options"].items():
            formatted += f"  {key}: {value}\n"

    formatted += "-" * 30 + "\n"

    # Add choice prompt
    one_meaning = mapping["1"]
    two_meaning = mapping["2"]
    formatted += f"Choices:\n  1: {one_meaning}\n  2: {two_meaning}\n"
    formatted += "Respond ONLY with '1' or '2'.\n"
    formatted += "Your choice ('1' or '2'):"

    options = ANSWER_OR_DELEGATE_OPTIONS

    if use_chat_template:
        try:
            messages = [
                {"role": "system", "content": ANSWER_OR_DELEGATE_SYSPROMPT + ANSWER_OR_DELEGATE_SETUP},
                {"role": "user", "content": formatted}
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            full_prompt = f"{ANSWER_OR_DELEGATE_SYSPROMPT}\n{ANSWER_OR_DELEGATE_SETUP}\n\n{formatted}"
    else:
        full_prompt = f"{ANSWER_OR_DELEGATE_SYSPROMPT}\n{ANSWER_OR_DELEGATE_SETUP}\n\n{formatted}"

    return full_prompt, options, mapping


def get_answer_or_delegate_signal(probs: np.ndarray, mapping: Dict[str, str]) -> float:
    """
    Convert answer/delegate probabilities to P(Answer).

    Higher values = more likely to answer (= more confident).
    This aligns with stated confidence signal (higher = more confident).
    """
    # probs[0] = P("1"), probs[1] = P("2")
    if mapping["1"] == "Answer":
        return float(probs[0])  # P(Answer) = P("1")
    else:
        return float(probs[1])  # P(Answer) = P("2")


def get_answer_or_delegate_response(probs: np.ndarray, mapping: Dict[str, str]) -> str:
    """Get the action (Answer or Delegate) based on argmax."""
    digit = "1" if probs[0] > probs[1] else "2"
    return mapping[digit]


# ============================================================================
# UNIFIED RESPONSE TO CONFIDENCE CONVERSION
# ============================================================================

def response_to_confidence(
    response: str,
    probs: np.ndarray = None,
    mapping: Dict[str, str] = None,
    task_type: str = "confidence"
) -> float:
    """
    Convert a meta response to a confidence value.

    For confidence task: Uses STATED_CONFIDENCE_MIDPOINTS lookup
    For delegate task: Uses P(Answer) from the probability distribution,
                       accounting for alternating mapping

    Args:
        response: The model's response ("1", "2", or S-Z for confidence)
        probs: Probability array [P("1"), P("2")] for delegate, or [P(S)...P(Z)] for confidence
        mapping: For delegate task, the mapping {"1": "Answer"/"Delegate", "2": ...}
        task_type: "confidence" or "delegate"
    """
    if task_type == "delegate":
        # For delegate task, confidence = P(Answer)
        # Need to account for alternating mapping
        if probs is not None and len(probs) >= 2 and mapping is not None:
            # Find which option corresponds to "Answer"
            if mapping.get("1") == "Answer":
                return float(probs[0])  # P("1") = P(Answer)
            else:
                return float(probs[1])  # P("2") = P(Answer)
        elif probs is not None and len(probs) >= 1:
            # Fallback: assume position 0 is Answer (old behavior)
            return float(probs[0])
        # Fallback if only response is known (no probs)
        if mapping is not None:
            return 1.0 if mapping.get(response) == "Answer" else 0.0
        return 1.0 if response == "1" else 0.0
    else:
        # For confidence task, use the midpoint lookup
        return STATED_CONFIDENCE_MIDPOINTS.get(response, 0.5)


# ============================================================================
# UNIFIED TASK INTERFACE
# ============================================================================

META_TASKS = {
    "stated_confidence": {
        "name": "Stated Confidence",
        "description": "Rate confidence on S-Z scale",
        "setup_prompt": STATED_CONFIDENCE_SETUP,
        "options": STATED_CONFIDENCE_OPTIONS,
        "option_midpoints": STATED_CONFIDENCE_MIDPOINTS,
        "format_prompt": format_stated_confidence_prompt,
        "get_signal": get_stated_confidence_signal,
        "get_response": get_stated_confidence_response,
        "signal_interpretation": "Expected confidence (0-1)",
    },
    "answer_or_delegate": {
        "name": "Answer or Delegate",
        "description": "Binary choice to answer or delegate",
        "setup_prompt": ANSWER_OR_DELEGATE_SETUP,
        "options": ANSWER_OR_DELEGATE_OPTIONS_ANSWER_FIRST,  # default
        "format_prompt": format_answer_or_delegate_prompt,
        "get_signal": get_answer_or_delegate_signal,
        "get_response": get_answer_or_delegate_response,
        "signal_interpretation": "P(Answer) - probability of choosing to answer",
    }
}

DIRECT_TASK = {
    "name": "Direct Multiple Choice",
    "description": "Answer a multiple choice question directly",
    "setup_prompt": MC_SETUP_PROMPT,
    "format_prompt": format_direct_prompt,
}


def get_meta_task(task_name: str) -> Dict:
    """Get a meta task configuration by name."""
    if task_name not in META_TASKS:
        raise ValueError(f"Unknown meta task: {task_name}. Available: {list(META_TASKS.keys())}")
    return META_TASKS[task_name]


def list_meta_tasks() -> List[str]:
    """List available meta task names."""
    return list(META_TASKS.keys())


# Convenience aliases for backward compatibility
# Maps old names used in scripts to new canonical names
META_OPTION_DICT = STATED_CONFIDENCE_OPTIONS
META_RANGE_MIDPOINTS = STATED_CONFIDENCE_MIDPOINTS
META_OPTIONS = list(STATED_CONFIDENCE_OPTIONS.keys())
META_SETUP_PROMPT = STATED_CONFIDENCE_SETUP
META_QUESTION_PROMPT = STATED_CONFIDENCE_QUESTION

DELEGATE_SETUP_PROMPT = ANSWER_OR_DELEGATE_SETUP
DELEGATE_SYSPROMPT = ANSWER_OR_DELEGATE_SYSPROMPT
DELEGATE_OPTIONS = ANSWER_OR_DELEGATE_OPTIONS

# Alias for the confidence task formatting (used as format_meta_prompt in scripts)
format_meta_prompt = format_stated_confidence_prompt
format_delegate_prompt = format_answer_or_delegate_prompt
