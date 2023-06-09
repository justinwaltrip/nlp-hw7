import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate as evaluate
from transformers import get_scheduler
import argparse
import subprocess
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration
from pathlib import Path

# TODO remove
tokenizer = None


def print_gpu_memory():
    """
    Print the amount of GPU memory used by the current process
    This is useful for debugging memory issues on the GPU
    """
    # check if gpu is available
    if torch.cuda.is_available():
        print(
            "torch.cuda.memory_allocated: %fGB"
            % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
        )
        print(
            "torch.cuda.memory_reserved: %fGB"
            % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
        )
        print(
            "torch.cuda.max_memory_reserved: %fGB"
            % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
        )

        p = subprocess.check_output("nvidia-smi")
        print(p.decode("utf-8"))


class BoolQADataset(torch.utils.data.Dataset):
    """
    Dataset for the dataset of BoolQ questions and answers
    """

    def __init__(self, passages, questions, answers, tokenizer, max_len, include_gold_passage):
        self.passages = passages
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_gold_passage = include_gold_passage

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        """
        This function is called by the DataLoader to get an instance of the data
        :param index:
        :return:
        """
        passage = str(self.passages[index])
        question = self.questions[index]
        answer = self.answers[index]

        # this is input encoding for your model. Note, question comes first since we are doing question answering
        # and we don't wnt it to be truncated if the passage is too long
        if self.include_gold_passage:
            input_encoding = question + " [SEP] " + passage
        else:
            input_encoding = question

        input_dict = self.tokenizer(
            input_encoding, max_length=self.max_len, return_tensors="pt", padding="max_length",
        )

        label = "yes" if answer else "no"
        label_dict = self.tokenizer(
            text_target=label, return_tensors="pt",
        )

        return {
            "input_ids": input_dict["input_ids"],
            "attention_mask": input_dict["attention_mask"],
            "labels": label_dict["input_ids"],
            "targets": torch.tensor(
                answer, dtype=torch.long
            ),  # labels are the answers (yes/no)
        }


def evaluate_model(model, dataloader, device):
    """Evaluate a PyTorch Model
    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :return accuracy
    """
    # load metrics
    dev_accuracy = evaluate.load("accuracy")

    # turn model into evaluation mode
    model.eval()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # resize input_ids, attention_mask
        input_ids_reshaped = input_ids.reshape(-1, input_ids.shape[-1])
        attention_mask_reshaped = attention_mask.reshape(-1, attention_mask.shape[-1])
        labels_reshaped = labels.reshape(-1, labels.shape[-1])

        outputs = model(input_ids=input_ids_reshaped, attention_mask=attention_mask_reshaped, labels=labels_reshaped)

        # get logits for no, yes (use only first token in sequence)
        logits = outputs.logits.reshape(input_ids_reshaped.shape[0], -1, outputs.logits.shape[-1])
        (id_yes, id_no) = ids
        selected_logits = logits[:, 0, [id_no, id_yes]]
        probabilities = selected_logits.softmax(dim=1)
        predictions = torch.argmax(probabilities, dim=1).cpu()
        dev_accuracy.add_batch(predictions=predictions, references=batch["targets"])

    # compute and return metrics
    return dev_accuracy.compute()


def train(model, num_epochs, train_dataloader, validation_dataloader, device, lr, ids):
    """Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on
    :param float lr: learning rate
    :return None
    """

    # here, we use the AdamW optimizer. Use torch.optim.Adam.
    # instantiate it on the untrained model parameters with a learning rate of 5e-5
    print(" >>>>>>>>  Initializing optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    criterion = torch.nn.CrossEntropyLoss()

    train_accs = []

    for epoch in range(num_epochs):
        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        model.train()

        # load metrics
        train_accuracy = evaluate.load("accuracy")

        print(f"Epoch {epoch + 1} training:")

        correct = 0
        loss = 0

        for i, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # resize input_ids, attention_mask
            input_ids_reshaped = input_ids.reshape(-1, input_ids.shape[-1])
            attention_mask_reshaped = attention_mask.reshape(-1, attention_mask.shape[-1])
            labels_reshaped = labels.reshape(-1, labels.shape[-1])

            outputs = model(input_ids=input_ids_reshaped, attention_mask=attention_mask_reshaped, labels=labels_reshaped)

            model_loss = outputs.loss.sum()
            loss += model_loss.item()

            model_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # get logits for no, yes (use only first token in sequence)
            logits = outputs.logits.reshape(input_ids_reshaped.shape[0], -1, outputs.logits.shape[-1])
            (id_yes, id_no) = ids
            selected_logits = logits[:, 0, [id_no, id_yes]]
            probabilities = selected_logits.softmax(dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu()

            correct += (predictions == batch["targets"]).sum().item()

            # update metrics
            train_accuracy.add_batch(
                predictions=predictions, references=batch["targets"]
            )

        # print evaluation metrics
        print(f" ===> Epoch {epoch + 1}")
        print(f" - Average training metrics: accuracy={train_accuracy.compute()}")
        print(f" - Average training loss: {loss / len(train_dataloader)}")

        # normally, validation would be more useful when training for many epochs
        val_accuracy = evaluate_model(model, validation_dataloader, device)
        print(f" - Average validation metrics: accuracy={val_accuracy}")

        train_accs.append(correct / len(train_dataloader.dataset))

    return train_accs


def pre_process(model_name, batch_size, device, small_subset=False, include_gold_passage=False):
    # download dataset
    print("Loading the dataset ...")
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle(seed=1)  # shuffle the data

    print("Slicing the data...")
    if small_subset:
        # use this tiny subset for debugging the implementation
        small_subset_size = 16
        dataset_train_subset = dataset["train"][:small_subset_size]
        dataset_dev_subset = dataset["train"][:small_subset_size]
        dataset_test_subset = dataset["train"][:small_subset_size]
    else:
        # since the dataset does not come with any validation data,
        # split the training data into "train" and "dev"
        # for simplicity of training, train your model on 500 training instances
        dataset_train_subset = dataset["train"][:500]
        dataset_dev_subset = dataset["validation"][:500]
        dataset_test_subset = dataset["train"][8000:8500]

    # maximum length of the input; any input longer than this will be truncated
    # we had to do some pre-processing on the data to figure what is the length of most instances in the dataset
    max_len = 128

    print("Loading the tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # get token ids for yes and no
    id_yes = tokenizer("yes", add_special_tokens=False)["input_ids"][0]
    id_no = tokenizer("no", add_special_tokens=False)["input_ids"][0]

    print("Loding the data into DS...")
    train_dataset = BoolQADataset(
        passages=list(dataset_train_subset["passage"]),
        questions=list(dataset_train_subset["question"]),
        answers=list(dataset_train_subset["answer"]),
        tokenizer=tokenizer,
        max_len=max_len,
        include_gold_passage=include_gold_passage,
    )
    validation_dataset = BoolQADataset(
        passages=list(dataset_dev_subset["passage"]),
        questions=list(dataset_dev_subset["question"]),
        answers=list(dataset_dev_subset["answer"]),
        tokenizer=tokenizer,
        max_len=max_len,
        include_gold_passage=include_gold_passage,
    )
    test_dataset = BoolQADataset(
        passages=list(dataset_test_subset["passage"]),
        questions=list(dataset_test_subset["question"]),
        answers=list(dataset_test_subset["answer"]),
        tokenizer=tokenizer,
        max_len=max_len,
        include_gold_passage=include_gold_passage,
    )

    print(" >>>>>>>> Initializing the data loaders ... ")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # load retriever
    retriever = RagRetriever.from_pretrained(model_name, index_name="exact", use_dummy_dataset=True)

    # from Hugging Face (transformers), read their documentation to do this.
    print("Loading the model ...")
    pretrained_model = RagSequenceForGeneration.from_pretrained(model_name, retriever=retriever)

    print("Moving model to device ..." + str(device))
    pretrained_model.to(device)
    return pretrained_model, train_dataloader, validation_dataloader, test_dataloader, (id_yes, id_no)


# the entry point of the program
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--small_subset", type=bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--include_gold_passage", type=bool, default=False)

    args = parser.parse_args()
    print(f"Specified arguments: {args}")

    # load the data and models
    (
        pretrained_model,
        train_dataloader,
        validation_dataloader,
        test_dataloader,
        ids,
    ) = pre_process(args.model, args.batch_size, args.device, args.small_subset)

    print(" >>>>>>>>  Starting training ... ")
    train_accs = train(
        pretrained_model,
        args.num_epochs,
        train_dataloader,
        validation_dataloader,
        args.device,
        args.lr,
        ids,
    )

    # print the GPU memory usage just to make sure things are alright
    print_gpu_memory()

    val_accuracy = evaluate_model(pretrained_model, validation_dataloader, args.device)
    print(f" - Average DEV metrics: accuracy={val_accuracy}")

    test_accuracy = evaluate_model(pretrained_model, test_dataloader, args.device)
    print(f" - Average TEST metrics: accuracy={test_accuracy}")

    # plot of training accuracy as a function of training epochs
    save_path = Path(f"figures/{args.experiment}.jpg")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.plot(train_accs)
    clf = plt.gcf()
    clf.savefig(save_path)
    plt.show()
