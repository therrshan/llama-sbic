from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import model_config
import os
import torch

def fine_tune(model,
              tokenizer,
              dataset,
              lora_r,
              lora_alpha,
              lora_dropout,
              bias,
              task_type,
              per_device_train_batch_size,
              gradient_accumulation_steps,
              warmup_steps,
              max_steps,
              learning_rate,
              fp16,
              logging_steps,
              output_dir,
              optim):

    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    target_modules = model_config.find_all_linear_names(model)

    peft_config = peft_config.create_peft_config(lora_r, lora_alpha, target_modules, lora_dropout, bias, task_type)
    model = get_peft_model(model, peft_config)

    model_config.print_trainable_parameters(model)

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            logging_steps=logging_steps,
            output_dir=output_dir,
            optim=optim,
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    model.config.use_cache = False

    do_train = True

    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print(metrics)

    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)

    del model
    del trainer
    torch.cuda.empty_cache()
