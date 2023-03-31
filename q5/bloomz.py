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
        train_prompt += f'Q: {train[i]["question"]} P: {train[i]["passage"][:100]} T or F?: {train[i]["answer"]} '
    train_prompt = train_prompt[:-6]

    # get 100 samples for evaluation
    test = dataset["validation"][:100]

    correct = 0
    for i in tqdm(range(100)):
        test_prompt = f'Q: {test["question"][i]} P: {test["passage"][i]} T or F?:'
        prompt = f"{train_prompt[:3000]} {test_prompt}"

        response = query({"inputs": prompt})
        generated = response[0]["generated_text"][len(prompt) + 1:].split(" ")[0]

        # print answer
        print(f"Question: {test['question'][i]}")
        print(f"Answer: {test['answer'][i]}")
        print(f"Prediction: {generated}")

        # get token right after prompt
        pred = generated == "True"
        correct += pred == test["answer"][i]

    print(f"Accuracy: {correct / 100:.2%}")


if __name__ == "__main__":
    main()
