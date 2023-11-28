import argparse
import re

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig


def init_model(model_name: str):
    print("init model ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        # torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.generation_config = GenerationConfig.from_pretrained(
        model_name,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True,
    )
    return model, tokenizer


task_names = [
    "computer_network",
    "operating_system",
    "computer_architecture",
    "college_programming",
    "college_physics",
    "college_chemistry",
    "advanced_mathematics",
    "probability_and_statistics",
    "discrete_mathematics",
    "electrical_engineer",
    "metrology_engineer",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_chemistry",
    "high_school_biology",
    "middle_school_mathematics",
    "middle_school_biology",
    "middle_school_physics",
    "middle_school_chemistry",
    "veterinary_medicine",
    "college_economics",
    "business_administration",
    "marxism",
    "mao_zedong_thought",
    "education_science",
    "teacher_qualification",
    "high_school_politics",
    "high_school_geography",
    "middle_school_politics",
    "middle_school_geography",
    "modern_chinese_history",
    "ideological_and_moral_cultivation",
    "logic",
    "law",
    "chinese_language_and_literature",
    "art_studies",
    "professional_tour_guide",
    "legal_professional",
    "high_school_chinese",
    "high_school_history",
    "middle_school_history",
    "civil_servant",
    "sports_science",
    "plant_protection",
    "basic_medicine",
    "clinical_medicine",
    "urban_and_rural_planner",
    "accountant",
    "fire_engineer",
    "environmental_impact_assessment_engineer",
    "tax_accountant",
    "physician",
]


def test_ceval_exam(model_name: str):
    cors = []
    model, tokenizer = init_model(model_name)
    for task in tqdm(task_names):
        _cos = []
        ds = load_dataset(r"ceval/ceval-exam", task)
        # print(ds["val"][0])
        # dataloader = DataLoader(ds["val"], batch_size=1)
        prompt_template = """
        {测试题目}
        A. {选项A}
        B. {选项B}
        C. {选项C}
        D. {选项D}
        答案：
        """
        for question in tqdm(ds["val"], desc=task):
            content = prompt_template.format(
                测试题目=question["question"],
                选项A=question["A"],
                选项B=question["B"],
                选项C=question["C"],
                选项D=question["D"],
            )
            # print("用户：", content)
            messages = [{"role": "user", "content": content}]
            response = model.chat(tokenizer, messages)
            match = re.search(r"([A-D])", response)
            pred = match.group(1) if match else ""
            cor = pred == question["answer"]
            _cos.append(cor)

            # print("response:", response)
            # print(question["answer"], question["explanation"])
        weighted_acc = np.mean(_cos)
        print("{} 测试: {} 平均正确率: {:.3f}".format(task, len(cors), weighted_acc))
        cors.extend(_cos)
    weighted_acc = np.mean(cors)
    print("一共测试: {} 平均正确率: {:.3f}".format(len(cors), weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="D:\github\ml\Model\Baichuan2-7B-Chat-4bits"
    )
    args = parser.parse_args()
    test_ceval_exam(args.model_name)
