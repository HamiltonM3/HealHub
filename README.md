# HealHub 

Under Construction // Alpha Version
Goal: Chatbot designed to interact with users about their mental health.
General Description: Program utilizes GPT-2 and the huggingface "council-chat" database to allow a user to have a conversation with the trained model.  

### Purpose:
This code does the following:
1. Fine-tunes a GPT-2 model on the "counsel-chat" dataset.
2. Allows a user to have a conversation with the trained model.

### Description:

1. **Import Necessary Libraries**: We start by importing the required Python libraries. These are mainly from `torch` and `transformers` which help us in deep learning model training and data processing.

2. **Load Dataset**: The dataset from "nbertagnolli/counsel-chat" is loaded. This dataset contains question-answer pairs.

3. **Data Preparation**:
   - **Combine questions & answers**: A function (`combine_qa`) is used to concatenate the questions and answers into a single string. This is useful for training the model to understand the context of the answers.
   - **Tokenization**: Convert our text data into a format that GPT-2 can understand. This is done using the `GPT2Tokenizer`.
   - **Conversion to Dataset**: Convert the tokenized data into a dataset format for easy feeding into the model.

4. **Set Training Parameters**: Defines parameters like batch size, logging directory, number of training epochs, etc. using `TrainingArguments`.

5. **Initialize Trainer**: A `Trainer` is created which will handle the fine-tuning of the GPT-2 model with the provided dataset.

6. **Model Training**: Using the `trainer.train()`, we fine-tune the GPT-2 model on our dataset.

7. **Save the Model**: After training, the model is saved locally.

8. **Chat with the Model**:
   - Load the fine-tuned model.
   - Define a function (`generate_response`) to generate a response for a given user input.
   - Engage in an interactive chat where the user types a message, and the model provides a response. The conversation continues until the user decides to exit.

### How to Use:
Simply run the script. After the model is trained, you'll enter an interactive mode. Just type your message and get a response from the fine-tuned GPT-2 model. Type 'exit' or 'quit' to end the conversation.
