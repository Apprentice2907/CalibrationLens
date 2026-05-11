print("SCRIPT STARTED")

print("Import torch...")
import torch
print("Torch imported")

print("Import transformers...")
from transformers import AutoTokenizer
print("Transformers imported")

print("Import datasets...")
from datasets import load_dataset
print("Datasets imported")

print("Checking GPU...")
print(torch.cuda.is_available())

print("Getting device name...")
print(torch.cuda.get_device_name(0))

print("Loading TriviaQA...")
ds = load_dataset("trivia_qa", "rc", split="validation[:5]")
print(ds)

print("Loading ARC...")
ds2 = load_dataset("ai2_arc", "ARC-Challenge", split="test[:5]")
print(ds2)

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")

print("DONE")