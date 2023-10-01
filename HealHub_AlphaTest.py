import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset, Dataset

# Load the dataset
dataset = load_dataset("nbertagnolli/counsel-chat")

# Combine questions and answers for training data
def combine_qa(data):
    return [(str(q) + "\n" + str(a)) for q, a in zip(data['questionText'], data['answerText']) if q is not None and a is not None]

train_texts = combine_qa(dataset['train'])
val_texts = train_texts  # Using the same data for validation in this example

# Tokenization
model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=150, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=150, return_tensors="pt")

# Convert to Dataset
train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']})
val_dataset = Dataset.from_dict({'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask']})

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=200,
    save_steps=1000,
    output_dir='./gpt2_counsel_chat',
    overwrite_output_dir=True,
    num_train_epochs=3,  # Adjust as needed
)

# Initialize Trainer
trainer = Trainer(
    model=GPT2LMHeadModel.from_pretrained(model_name),
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Save the model
trainer.save_model()

# Chat with the model
model = GPT2LMHeadModel.from_pretrained('./gpt2_counsel_chat')
model.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_response(input_text, max_length=150):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        response_ids = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Counsel-Chat GPT-2: Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Counsel-Chat GPT-2: Goodbye!")
            break
        response = generate_response(user_input)
        print(f"Counsel-Chat GPT-2: {response}")
