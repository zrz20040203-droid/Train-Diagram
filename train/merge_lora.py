import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 原始模型
base_model_path = "/root/autodl-tmp/Qwen2.5-7B-Instruct"

# LoRA权重
lora_path = "/root/autodl-tmp/output/qwen2.5_lora"

# 合并后模型保存位置
save_path = "/root/autodl-tmp/Qwen2.5-7B-finetuned"

print("加载原始模型...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("加载LoRA权重...")
model = PeftModel.from_pretrained(model, lora_path)

print("开始合并LoRA...")
model = model.merge_and_unload()

print("保存模型...")
model.save_pretrained(save_path)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(save_path)

print("合并完成 ✅")