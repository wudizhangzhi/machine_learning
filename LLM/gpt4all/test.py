import os
import platform
import subprocess
from tempfile import NamedTemporaryFile

import torch
from colorama import Fore, Style
from gpt4all import GPT4All


def init_model():
    model =  GPT4All("orca-mini-3b-gguf2-q4_0.gguf")
    print("default system template:", repr(model.config['systemPrompt']))
    print("default prompt template:", repr(model.config['promptTemplate']))
    return model


def clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    print(
        Fore.YELLOW
        + Style.BRIGHT
        + "欢迎使用大模型，输入进行对话，vim 多行输入，clear 清空历史，CTRL+C 中断生成，stream 开关流式生成，exit 结束。"
    )
    return []


def vim_input():
    with NamedTemporaryFile() as tempfile:
        tempfile.close()
        subprocess.call(["vim", "+star", tempfile.name])
        text = open(tempfile.name).read()
    return text


def main(stream=True):
    model = init_model()
    messages = clear_screen()
    while True:
        prompt = input(Fore.GREEN + Style.BRIGHT + "\n用户：" + Style.NORMAL)
        if prompt.strip() == "exit":
            break
        if prompt.strip() == "clear":
            messages = clear_screen()
            continue
        if prompt.strip() == "vim":
            prompt = vim_input()
            print(prompt)
        print(Fore.CYAN + Style.BRIGHT + "\nBot：" + Style.NORMAL, end="")
        if prompt.strip() == "stream":
            stream = not stream
            print(
                Fore.YELLOW + "({}流式生成)\n".format("开启" if stream else "关闭"),
                end="",
            )
            continue
        # messages.append({"role": "user", "content": prompt})
        if stream:
            position = 0
            try:
                # system_template = ""
                # prompt_template = ""
                with model.chat_session(stream=True):
                    for token in model.generate(prompt):
                        
                # for response in model.chat(tokenizer, messages, stream=True):
                #     print(response[position:], end="", flush=True)
                #     position = len(response)
                #     if torch.backends.mps.is_available():
                #         torch.mps.empty_cache()
            except KeyboardInterrupt:
                pass
            print()
        else:
            response = model.chat(tokenizer, messages)
            print(response)
            # if torch.backends.mps.is_available():
            #     torch.mps.empty_cache()
        # messages.append({"role": "assistant", "content": response})
    print(Style.RESET_ALL)


if __name__ == "__main__":
    main()
