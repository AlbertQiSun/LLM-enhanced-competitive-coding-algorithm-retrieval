import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import json
import warnings
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

warnings.filterwarnings("ignore")

class Solution(BaseModel):
    possible_tags: str
    probability: float

class Annotation(BaseModel):
    possible_algorithm: list[Solution]

dataset = load_dataset("AlbertHatsuki/codeforces-question-solution-label")
possible_tags = sorted(list({tag for split in dataset.values() for example in split for tag in example["tags"]}))

# all_tags = sorted(list({tag for split in dataset.values() for example in split for tag in example["tags"]}))
py_dataset = dataset.filter(lambda example: example["language"] == "py")
java_dataset = dataset.filter(lambda example: example["language"] == "java")
cpp_dataset = dataset.filter(lambda example: example["language"] == "cpp")

py_tags = sorted(list({tag for split in py_dataset.values() for example in split for tag in example["tags"]}))
java_tags = sorted(list({tag for split in java_dataset.values() for example in split for tag in example["tags"]}))
cpp_tags = sorted(list({tag for split in cpp_dataset.values() for example in split for tag in example["tags"]}))

py_model_path = "./py/py_fine_tuned_unixcoder"
java_model_path = "./java/java_fine_tuned_unixcoder"
cpp_model_path = "./cpp/cpp_fine_tuned_unixcoder"
tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')

py_model = AutoModelForSequenceClassification.from_pretrained(py_model_path)
java_model = AutoModelForSequenceClassification.from_pretrained(java_model_path)
cpp_model = AutoModelForSequenceClassification.from_pretrained(cpp_model_path)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
py_model.to(device)
java_model.to(device)
cpp_model.to(device)

def read_data():
    with open('earliest_accepted_code.json', 'r', encoding='utf-8') as f:
        data = json.load(f)['RECORDS']
    return data

def labeling(code, lang):
    if lang == 'java':
        model = java_model
        all_tags = java_tags
    elif lang in ['py2', 'python', 'pypy3']:
        model = py_model
        all_tags = py_tags
    else:
        model = cpp_model
        all_tags = cpp_tags
    inputs = tokenizer(
        code,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
        logits = output.logits
    probabilities = torch.sigmoid(logits).cpu().numpy()
    probabilities = probabilities[0]
    original_tags = {}
    for i in range(len(probabilities)):
        original_tags[all_tags[i]] = probabilities[i]
    sorted_tags = sorted(original_tags.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_tags[:5])

def get_prompt(description):
    default_setting = (
        f"You are a highly skilled coding assistant specialized in identifying the most suitable algorithms for solving International Collegiate Programming Contest (ICPC) problems.\n"
        f"You must strictly adhere to the following rules when responding to my questions:\n"
        f"**1.** I will provide a complete problem description, including the title, description, input, output, time limit, and memory limit. You must only select algorithms from the list of possible algorithms I provide in the appendix. No other algorithms are allowed.\n"
        f"**2.** Carefully analyze the problem description, input, and output instructions, as they contain critical details for solving the problem. Pay special attention to any LaTeX-formatted equations, as they are key hints for determining the solution.\n"
        f"**3.** When selecting algorithms, consider the time limit, memory limit, and input range to evaluate time complexity. Assume a baseline computation speed of $1 \times 10^9$ operations per second. Choose algorithms that satisfy these constraints.\n"
        f"**4.** You may only select the top 2-3 most suitable algorithms as solutions. Do not exceed this limit.\n"
        f"**5.** For each selected algorithm, use the exact name from the list in the appendix. Do not invent or modify algorithm names.\n"
        f"**6.** For each algorithm, provide a probability of it being the correct solution. Use a sigmoid function for probability distribution and avoid overly optimistic estimates.\n"
        f"#Appendix:\n"
        f"List of allowed algorithm tags (you must choose only from these):\n"
        f"{possible_tags}"
    )
    prompt = (
        f'Here is the question description, please strictly follow the rules in the system input to respond me.\n'
        f'{description}'
    )
    return default_setting, prompt

def gpt_labeling(description):
    default_setting, prompt = get_prompt(description)
    response = client.beta.chat.completions.parse(
        model='gpt-4o-mini',
        messages=[
            {
                'role': 'system',
                'content': default_setting,
            },
            {
                'role': 'user',
                'content': prompt
            },
        ],
        response_format=Annotation,
    )
    return dict(response.choices[0].message.parsed)['possible_algorithm']

def calculating_prediction(fast_prediction, early_prediction, response):
    overall = {}
    try:
        for keys in list(fast_prediction.keys()):
            overall[keys] = fast_prediction[keys] * 0.25
        for keys in list(early_prediction.keys()):
            overall[keys] = overall.get(keys, 0) + early_prediction[keys] * 0.25
    finally:
        for item in response:
            item = dict(item)
            overall[item['possible_tags']] = overall.get(item['possible_tags'], 0) + item['probability'] * 0.05
    final_sorted = sorted(overall.items(), key=lambda x: x[1], reverse=True)
    return final_sorted[:3]

def run():
    final_data = []
    with open('./earliest_accepted_code.json', 'r', encoding='utf-8') as f:
        earliest_data = json.load(f)['RECORDS']
    with open('./fastest_accepted_code.json', 'r', encoding='utf-8') as f:
        fastest_data = json.load(f)['RECORDS']
    with open('./problem_description.json', 'r', encoding='utf-8') as f:
        problem_description = json.load(f)
    description_list = list(problem_description.keys())
    fastest_pointer = 0
    for earliest_pointer in tqdm(range(len(earliest_data)), desc='Predicting and merging', total=len(earliest_data)):
        earliest_item = earliest_data[earliest_pointer]
        fastest_item = fastest_data[fastest_pointer]
        earliest_id = earliest_item['problem_id']
        fastest_id = fastest_item['problem_id']
        if earliest_id == fastest_id:
            fast_prediction = labeling(fastest_item['code'], fastest_item['lang'])
            early_prediction = labeling(earliest_item['code'], earliest_item['lang'])
            fastest_pointer += 1
            description = json.dumps(problem_description[earliest_id])
            response = gpt_labeling(description)
            final_prediction = calculating_prediction(fast_prediction, early_prediction, response)
            final_data.append({'problem_id': earliest_id, 'problem_description': description, 'prediction': final_prediction})
            try:
                description_list.remove(earliest_id)
            except Exception:
                pass
        elif earliest_id > fastest_id:
            while fastest_id < earliest_id:
                fast_prediction = labeling(fastest_item['code'], fastest_item['lang'])
                early_prediction = {}
                description = json.dumps(problem_description[earliest_id])
                response = gpt_labeling(description)
                final_prediction = calculating_prediction(fast_prediction, early_prediction, response)
                final_data.append({'problem_id': fastest_id, 'problem_description': description, 'prediction': final_prediction})
                try:
                    description_list.remove(fastest_id)
                except Exception:
                    pass
                fastest_pointer += 1
                fastest_item = fastest_data[fastest_pointer]
                fastest_id = fastest_item['problem_id']
            if earliest_id == fastest_id:
                fast_prediction = labeling(fastest_item['code'], fastest_item['lang'])
                early_prediction = labeling(earliest_item['code'], earliest_item['lang'])
                fastest_pointer += 1
                description = json.dumps(problem_description[earliest_id])
                response = gpt_labeling(description)
                final_prediction = calculating_prediction(fast_prediction, early_prediction, response)
                final_data.append({'problem_id': earliest_id, 'problem_description': description, 'prediction': final_prediction})
                try:
                    description_list.remove(earliest_id)
                except Exception:
                    pass
        else:
            fast_prediction = {}
            early_prediction = labeling(earliest_item['code'], fastest_item['lang'])
            description = json.dumps(problem_description[earliest_id])
            response = gpt_labeling(description)
            final_prediction = calculating_prediction(fast_prediction, early_prediction, response)
            final_data.append({'problem_id': earliest_id, 'problem_description': description, 'prediction': final_prediction})
            try:
                description_list.remove(earliest_id)
            except Exception:
                pass
    for id in tqdm(description_list):
        description_data = problem_description[id]
        response = gpt_labeling(description_data)
        fast_prediction = {}
        early_prediction = {}
        final_prediction = calculating_prediction(fast_prediction, early_prediction, response)
        try:
            final_data.append({'problem_id': id, 'problem_description': json.loads(description_data), 'prediction': final_prediction})
        except Exception:
            final_data.append({'problem_id': id, 'problem_description': description_data, 'prediction': final_prediction})
    sorted_final_data = sorted(final_data, key=lambda x: int(x["problem_id"]))
    with open('./final_data.json', 'w', encoding='utf-8') as f:
        json.dump(sorted_final_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    run()