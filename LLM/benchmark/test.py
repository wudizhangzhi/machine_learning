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


def test_ceval_exam(model_name: str):
    cors = []
    ds = load_dataset(r"ceval/ceval-exam", "computer_network")
    # print(ds["val"][0])
    # dataloader = DataLoader(ds["val"], batch_size=1)
    model, tokenizer = init_model(model_name)
    prompt_template = """
    {测试题目}
    A. {选项A}
    B. {选项B}
    C. {选项C}
    D. {选项D}
    答案：
    """
    for question in tqdm(ds["val"]):
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
        cors.append(cor)

        # print("response:", response)
        # print(question["answer"], question["explanation"])
    weighted_acc = np.mean(cors)
    print("一共测试: {} 平均正确率: {:.3f}".format(len(cors), weighted_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="D:\github\ml\Model\Baichuan2-7B-Chat-4bits"
    )
    args = parser.parse_args()
    test_ceval_exam(args.model_name)
