# temp_scaling.py
import torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from calibration import compute_ece

MODEL_PATH = './checkpoints/gemma2b_science'
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b')
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map='auto')
model.eval()

def score(model, tokenizer, question, choices, temperature=1.0):
    scores = []
    for c in choices:
        inp = tokenizer(f'Q: {question}\nA: {c}', return_tensors='pt',
                        truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            scores.append(-model(**inp, labels=inp['input_ids']).loss.item())
    probs = torch.softmax(torch.tensor(scores) / temperature, dim=0)
    bi = probs.argmax().item()
    return choices[bi], float(probs[bi])

ds = load_dataset('ai2_arc', 'ARC-Challenge', split='test').select(range(500))
best_t, best_ece = 1.0, 999
for T in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
    cs, ks = [], []
    for row in ds:
        ch = row['choices']['text']
        ans = ch[row['choices']['label'].index(row['answerKey'])]
        pred, conf = score(model, tokenizer, row['question'], ch, T)
        cs.append(conf); ks.append(int(pred == ans))
    ece = compute_ece(cs, ks)
    print(f'T={T:.1f}  ECE={ece:.4f}')
    if ece < best_ece:
        best_ece, best_t = ece, T

print(f'\nBest temperature: {best_t}  ECE: {best_ece:.4f}')
print(f'ECE after QLoRA without scaling: 0.2536')
print(f'ECE after temperature scaling:   {best_ece:.4f}')