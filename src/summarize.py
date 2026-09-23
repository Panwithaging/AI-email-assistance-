from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from preprocess import preprocessing

import torch

from models import qwen1_tokenizer,qwen1_model


def summarizer(text):
    prompt = f"""
    You are an AI email assistant.
    Read the email and generate a short notification for the recipient.
    Requirements:
    - Write in second person.
    - Start with "You".
    - Use at most 2 sentences.
    - Keep the notification under 40 words.
    - Include only the most important information from the email.
    - If present, include:
    - the purpose of the email,
    - important date or time,
    - important deadline,
    - required action.
    - Omit greetings, signatures, contact information, background details, compliments, promotional text, and unnecessary explanations.
    - Do not copy long sentences from the email.
    - Do not invent information.
    - Return only the notification.
    Email:
    {text}"""

    message=[
        {
            "role":"user",
            "content":prompt
        }
    ]

    formated_message=qwen1_tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs=qwen1_tokenizer(
        formated_message,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        output=qwen1_model.generate(inputs["input_ids"],
                              attention_mask=inputs["attention_mask"],
                              max_new_tokens=80,
                              num_beams=7,
                              early_stopping=True
                              )
    generate_token=output[0][inputs["input_ids"].shape[1]:]
    summary=qwen1_tokenizer.decode(
        generate_token,
        skip_special_tokens=True
    )
    summary=summary.strip()
    if summary.startswith("Dear"):
        lines=summary.split("\n")
        summary="\n".join(lines[1:]).strip()

    return summary





