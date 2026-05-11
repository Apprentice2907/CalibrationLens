import torch, pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from calibration import full_report
from tqdm import tqdm

MODEL_PATH = 'D:\\hf_cache\\gemma2b'   # before: base model | after: checkpoint path
PHASE      = 'before'              # 'before' or 'after'
N          = 500

def score(model, tokenizer, question, choices):
    scores = []
    for c in choices:
        inp = tokenizer(f'Q: {question}\nA: {c}', return_tensors='pt',
                        truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            scores.append(-model(**inp, labels=inp['input_ids']).loss.item())
    probs = torch.softmax(torch.tensor(scores), dim=0)
    bi = probs.argmax().item()
    return choices[bi], float(probs[bi])

def eval_trivia(model, tok):
    ds = load_dataset('trivia_qa','rc',split='validation').select(range(N))
    cs,ks = [],[]
    for row in tqdm(ds, desc='TriviaQA'):
        ans = row['answer']['value']
        choices = [ans,'I do not know','Cannot determine','Other']
        pred,conf = score(model,tok,row['question'],choices)
        cs.append(conf); ks.append(int(pred==ans))
    return full_report(cs, ks, f'TriviaQA_{PHASE}')

def eval_arc(model, tok):
    ds = load_dataset('ai2_arc','ARC-Challenge',split='test').select(range(N))
    cs,ks = [],[]
    for row in tqdm(ds, desc='ARC'):
        ch = row['choices']['text']; lb = row['choices']['label']
        ans = ch[lb.index(row['answerKey'])]
        pred,conf = score(model,tok,row['question'],ch)
        cs.append(conf); ks.append(int(pred==ans))
    return full_report(cs, ks, f'ARC_{PHASE}')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_4bit=True, device_map='auto')
model.eval()

results = [eval_trivia(model,tokenizer), eval_arc(model,tokenizer)]
pd.DataFrame(results).to_csv(f'results_{PHASE}.csv', index=False)
print(f'Saved: results_{PHASE}.csv')
