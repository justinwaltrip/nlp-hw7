import torch
from datasets import load_dataset
import os
import openai
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv("config.env")

openai.api_key = os.getenv("OPENAI_API_KEY")


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

    prompt = ""
    for i in range(8):
        prompt += f'Q: {train[i]["question"]} P: {train[i]["passage"]} T or F?: {train[i]["answer"]} '

    # get 30 samples for evaluation
    test = dataset["validation"][:30]

    correct = 0
    for i in tqdm(range(30)):
        test_prompt = prompt + f'Q: {test["question"][i]} P: {test["passage"][i]} T or F?:'

        response = openai.Completion.create(
            model="davinci",
            prompt=test_prompt,
            temperature=0.7,
            max_tokens=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        print(f"Question: {test['question'][i]}")
        print(f"Answer: {test['answer'][i]}")
        print(f"Prediction: {response['choices'][0]['text'].strip()}")

        pred = response["choices"][0]["text"].strip() == "True"
        correct += pred == test["answer"][i]

    print(f"Accuracy: {correct / 30:.2%}")


if __name__ == "__main__":
    main()
