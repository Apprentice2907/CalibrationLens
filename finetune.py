# finetune.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer

MODEL  = 'google/gemma-2-2b'
OUT = './checkpoints/gemma2b_r32'

model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(r=32, lora_alpha=64,
    target_modules=['q_proj','k_proj','v_proj','o_proj'],
    lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

ds = load_dataset('ai2_arc', 'ARC-Challenge', split='train')
ds = ds.select(range(min(2000, len(ds))))

def fmt(row):
    return {'text': f"Q: {row['question']}\nA: {row['choices']['text'][row['choices']['label'].index(row['answerKey'])]}"}

ds = ds.map(fmt)

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds,
    dataset_text_field='text', max_seq_length=256,
    args=TrainingArguments(
        output_dir=OUT, num_train_epochs=3,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-4, warmup_steps=50,
        fp16=True, optim='adamw_torch',
        logging_steps=50, save_steps=200))

trainer.train()
model.save_pretrained(OUT)
print('Done. r=8 checkpoint saved.')