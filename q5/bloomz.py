import torch
from datasets import load_dataset
import os
from dotenv import load_dotenv
from tqdm import tqdm
import requests

load_dotenv("config.env")

API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"
headers = {"Authorization": f"Bearer {API_TOKEN}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def main():
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle(seed=1)  # shuffle the data

    # get 4 samples where the answer is true
    train_true = dataset["train"].filter(lambda x: x["answer"]).select(range(4))
    train_false = dataset["train"].filter(lambda x: not x["answer"]).select(range(4))

    # combine the two datasets with alternating samples
    train = []
    for i in range(4):
        train.append(train_true[i])
        train.append(train_false[i])

    train_prompt = ""
    for i in range(8):
        train_prompt += f'{train[i]["question"]}? [SEP] {train[i]["passage"]} [SEP] {train[i]["answer"]} [SEP] '
    train_prompt = train_prompt[:-6]

    # get 30 samples for evaluation
    test = dataset["validation"][:100]

    correct = 0
    for i in tqdm(range(100)):
        test_prompt = f'[SEP] {test["question"][i]}? [SEP] {test["passage"][i]} [SEP]'
        prompt = f'{train_prompt[:3000]} {test_prompt}'

        response = query({"inputs": prompt})

        # print answer
        print(f"Question: {test['question'][i]}")
        print(f"Answer: {test['answer'][i]}")
        print(f"Prediction: {response[0]['generated_text'].split(' ')[-1]}")

        pred = response[0]["generated_text"].split(" ")[-1] == "True"
        correct += pred == test["answer"][i]

    print(f"Accuracy: {correct / 30:.2%}")


if __name__ == "__main__":
    main()
