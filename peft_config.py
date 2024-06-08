from peft import LoraConfig

def create_peft_config(r, lora_alpha, target_modules, lora_dropout, bias, task_type):

    config = LoraConfig(
        r = r,
        lora_alpha = lora_alpha,
        target_modules = target_modules,
        lora_dropout = lora_dropout,
        bias = bias,
        task_type = task_type,
    )

    return config