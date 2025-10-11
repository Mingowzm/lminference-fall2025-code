import json
import re
import os
import time
from typing import Dict, Any, List
from tqdm import tqdm

import litellm
from openai import OpenAI
from datasets import load_dataset


choices = ["A", "B", "C", "D"]

SYS_MSG ="Based on the provided Input (if any) and Generated Text, answer the ensuing Questions with either a YES or NO choice. Your selection should be based on your judgment as well as the following rules:\n\n- YES: Select 'YES' if the generated text entirely fulfills the condition specified in the question. However, note that even minor inaccuracies exclude the text from receiving a 'YES' rating. As an illustration. consider a question that asks. \"Does each sentence in the generated text use a second person?\" If even one sentence does not use the second person, the answer should NOT be 'YES'. To qualify for a 'YES' rating, the generated text must be entirely accurate and relevant to the question\n\n- NO: Opt for 'NO' if the generated text fails to meet the question's requirements or provides no information that could be utilized to answer the question. For instance, if the question asks. \"Is the second sentence in the generated text a compound sentence?\" and the generated text only has one sentence. it offers no relevant information to answer the question. Consequently, the answer should be 'NO'.'''"

# Define decoding strategies
decoding_strategies = {
    "default": {},
    "greedy": {"do_sample": False},
    "temp_0.25": {"do_sample": True, "temperature": 0.25},
    "temp_1.5": {"do_sample": True, "temperature": 1.5},
    "beam_3": {"num_beams": 3},
    "beam_25": {"num_beams": 25},
    "typical": {"do_sample": True, "typical_p": 0.9}
}

def load_custom_dataset(dataset_name: str):
    # Load datasets from the new source
    if dataset_name == "graph":
        raw_dataset = load_dataset("vashistht/11763_datasets", "graph_dev")
        dataset = list(raw_dataset['dev_test'])
    elif dataset_name == "infobench":
        raw_dataset = load_dataset("vashistht/11763_datasets", "infobench")
        dataset = list(raw_dataset['dev_test'])
    elif dataset_name == "mmlu_med":
        raw_dataset = load_dataset("vashistht/11763_datasets", "mmlu_med")
        dataset = list(raw_dataset['dev_test'])
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return dataset


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(example, include_answer=False):
    prompt = f"Question: {example['question']}\n Options:"
    these_choices = example["choices"]

    for i in range(len(these_choices)):
        prompt += f"\n{choices[i]}. {these_choices[i]}"

    prompt += "\nAnswer:"
    if include_answer:
        # for in-context learning
        prompt += f" {choices[example['answer']]}\n\n"
    return prompt


def extract_answer(text):
    # remove the latex box, common for AIME
    text = re.sub(r'\$\\boxed\{([A-Za-z])\}\$', r'\1', text)

    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        pattern = r"option \(?([A-J])\)?"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return None


def generate_problem_prompt(dataset_name: str, example: str) -> str:
    if dataset_name == "mmlu_med":
        # https://github.com/hendrycks/test/blob/master/evaluate.py
        subject = example.get('subject', 'medicine')  # Default to medicine if no subject
        prompt = f"The following is a multiple choice question (with answers) about {format_subject(subject)}.  Output the answer in the format of \"The answer is (X)\" at the end.\n\n"
        return prompt + format_example(example, include_answer=False)

    elif dataset_name == "infobench":
        # https://arxiv.org/pdf/2401.03601
        return f"Instruction: {example['instruction']}\nQuestion: {example['input']}\nGeneration:"

    elif dataset_name == "graph":
        # Assume graph dataset has similar structure to MMLU
        if 'instruction' in example and 'input' in example:
            return f"Instruction: {example['instruction']}\nQuestion: {example['input']}\nGeneration:"
        else:
            return format_example(example, include_answer=False)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def query_llm(prompt: str, model: str, api_key: str, api_base: str, decoding_strategy: str = "greedy", prompt_mode=True) -> Dict[str, Any]:
    """
    Query a language model with the given prompt and decoding strategy.

    Args:
        prompt: The problem description
        model: Model name
        api_key: API key for the model
        api_base: Base URL for the API
        decoding_strategy: Name of the decoding strategy to use
        prompt_mode: Whether to use prompt mode or message mode

    Returns:
        String containing the model response
    """

    try:
        # Get decoding parameters
        decoding_params = decoding_strategies.get(decoding_strategy)

        # Prepare completion arguments
        if prompt_mode:
            completion_args = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "api_base": api_base,
                "api_key": api_key,
                "max_tokens": 2000,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        else:
            # prompts are formatted messages already
            completion_args = {
                "model": model,
                "messages": prompt,
                "api_base": api_base,
                "api_key": api_key,
                "max_tokens": 2000,
                "chat_template_kwargs": {"enable_thinking": False}
            }

        # Add decoding strategy parameters
        completion_args.update(decoding_params)

        response = litellm.completion(**completion_args)
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error querying LLM with {decoding_strategy}: {e}")
        return None


def convert_llm_response_to_solution(llm_response: str, dataset_name: str) -> str:
    if dataset_name == "mmlu_med":
        # adapted from https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_api.py
        return extract_answer(llm_response.replace('**', ''))
    elif dataset_name == "infobench":
        return llm_response
    elif dataset_name == "graph":
        # Assume graph dataset uses similar format to MMLU
        return extract_answer(llm_response.replace('**', ''))
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def bool_ratio(bool_results: List[bool]) -> float:
    "Calculate true false ratio for eval results"
    count = {"true":0, "false":0}
    for entry in bool_results:
        if entry:
            count["true"] += 1
        else:
            count["false"] += 1

    return count['true']/sum(count.values())


def info_bench_eval(example: str, predicted_solution: str, model: str, api_key: str, openai_api_key: str) -> float:
    # https://github.com/qinyiwei/InfoBench/blob/main/evaluation.py
    message = []
    answer = ""
    input_task = example['input']
    output = predicted_solution
    client = OpenAI(api_key=openai_api_key)

    for question in example["decomposed_questions"]:
        if len(message) == 0:
            if input_task:
                content =  f"{SYS_MSG}\n\nInput:\n\"{input_task}\"\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
            else:
                content =  f"{SYS_MSG}\n\nGenerated Text:\n\"{output}\"\n\nQuestion:\n{question}\n"
        else:
            content = f"{question}\n"
        message.append({"role": "user", "content": content})
        # create a chat completion
        success = False
        early_stop = False
        while not success:
            try:
                # default config
                temperature = 1.0
                eval_model = "gpt-4o-mini"  # Changed to available model

                completion = client.chat.completions.create(
                        model=eval_model,
                        messages=message,
                        temperature=temperature,
                    )
                generation = completion.choices[0].message.content
                message.append(
                        {"role": "assistant", "content": generation})
                # check if generation is yes or no
                if generation.lower().startswith("yes") or generation.lower().startswith("no"):
                    if generation.lower().startswith("yes"):
                        answer += "Yes\n"
                    else:
                        answer += "No\n"
                else:
                    if "YES" in generation and "NO" not in generation:
                        answer += "Yes\n"
                    elif "YES" not in generation and "NO" in generation:
                        answer += "No\n"
                    else:
                        for msg in message:
                            print(msg['content'])
                        print("NO YES or NO answer!" + generation)
                        answer += "None\n"
                        early_stop = True
                        break
                success = True
            except Exception as e:
                print("ERROR!")
                print(e)
                print("Retry!")
                time.sleep(5)

            # when no answer occurs, break the loop and continue to next instance
            if early_stop:
                break

    answer = answer[:-1]
    # save eval results as List[bool]
    bool_results = []
    for i in answer.split('\n'):
        if i == "Yes":
            bool_results.append(True)
        elif i == "No":
            bool_results.append(False)
        else:
            bool_results.append(None)

    return bool_ratio(bool_results)


def evaluate_solution(example: str, predicted_solution: str, dataset_name: str, model: str, api_key: str, openai_api_key: str="") -> float:
    if dataset_name == "mmlu_med":
        return choices[example["answer"]] == predicted_solution
    elif dataset_name == "infobench":
        # https://github.com/qinyiwei/InfoBench/blob/main/evaluation.py
        return info_bench_eval(example, predicted_solution, model, api_key, openai_api_key)
    elif dataset_name == "graph":
        # Assume graph dataset has answer field like MMLU
        if "answer" in example:
            return choices[example["answer"]] == predicted_solution
        else:
            # If no answer field, assume it's like InfoBench
            return info_bench_eval(example, predicted_solution, model, api_key, openai_api_key)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def run_evaluation(examples: List[Dict[str, Any]], model: str, api_key: str, api_base: str,
                   task: str, decoding_strategy: str = "greedy", openai_api_key: str="") -> Dict[str, Any]:
    """
    Run evaluation on a list of examples with a specific decoding strategy.

    Args:
        examples: List of example dictionaries
        model: Model name
        api_key: API key
        api_base: Base URL for the API
        task: Task name
        decoding_strategy: Decoding strategy to use
        openai_api_key: OpenAI API key for InfoBench evaluation

    Returns:
        Dictionary containing evaluation results
    """
    results = []
    total_score = 0.0

    for i, example in tqdm(enumerate(examples, 1), total=len(examples), desc=f"Evaluating with {decoding_strategy}"):
        prompt = generate_problem_prompt(task, example)
        llm_response = query_llm(prompt, model, api_key, api_base, decoding_strategy)

        if llm_response is None:
            print(f"Failed to get response for example {i}")
            continue

        predicted_solution = convert_llm_response_to_solution(llm_response, task)

        # Evaluate
        score = evaluate_solution(example, predicted_solution, task, model, api_key, openai_api_key)

        total_score += score

        results.append({
            "example_id": i,
            "example": example,
            "llm_response": llm_response,
            "predicted_solution": predicted_solution,
            "score": score,
            "decoding_strategy": decoding_strategy
        })

    average_score = total_score / len(examples) if examples else 0.0

    return {
        "model": model,
        "decoding_strategy": decoding_strategy,
        "average_score": average_score,
        "total_examples": len(examples),
        "results": results
    }


def run_full_evaluation(examples: List[Dict[str, Any]], model: str, api_key: str, api_base: str,
                        task: str, strategies_to_test: List[str] = None, openai_api_key: str = "") -> Dict[str, Any]:
    """
    Run evaluation across multiple decoding strategies.

    Args:
        examples: List of example dictionaries
        model: Model name
        api_key: API key
        api_base: Base URL for the API
        task: Task name
        strategies_to_test: List of decoding strategies to test (default: all)
        openai_api_key: OpenAI API key for InfoBench evaluation

    Returns:
        Dictionary containing results for all strategies
    """
    if strategies_to_test is None:
        strategies_to_test = list(decoding_strategies.keys())

    all_results = {}

    for strategy in strategies_to_test:
        print(f"\n=== Running evaluation with {strategy} strategy ===")
        strategy_results = run_evaluation(examples, model, api_key, api_base, task, strategy, openai_api_key)
        all_results[strategy] = strategy_results
        print(f"Average score for {strategy}: {strategy_results['average_score']:.4f}")

    return all_results