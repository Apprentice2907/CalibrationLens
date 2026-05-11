# finetune.py
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL   = 'google/gemma-2-2b'
DOMAIN  = 'science'   # 'science' or 'general'
OUT     = f'./checkpoints/gemma2b_{DOMAIN}'

model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=256, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model,
    r=16, lora_alpha=32, target_modules=['q_proj','k_proj','v_proj','o_proj'],
    lora_dropout=0.05, bias='none', use_gradient_checkpointing=True)

ds = load_dataset('ai2_arc','ARC-Challenge',split='train') if DOMAIN=='science' \
     else load_dataset('trivia_qa','rc',split='train')
ds = ds.select(range(min(2000,len(ds))))

SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, max_seq_length=256,
    args=TrainingArguments(output_dir=OUT, num_train_epochs=3,
        per_device_train_batch_size=4, gradient_accumulation_steps=4,
        learning_rate=2e-4, warmup_steps=50, fp16=True, optim='adamw_8bit',
        logging_steps=50, save_steps=200)).train()

model.save_pretrained(OUT)
print(f'Done. Now set MODEL_PATH={OUT} and PHASE=after in evaluate.py')
