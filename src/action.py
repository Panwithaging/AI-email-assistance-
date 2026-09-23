from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from preprocess import preprocessing

import torch

from models import qwen1_tokenizer ,qwen1_model


def require_action(text):
    prompt=f"""You are an AI email assistant.
    Read the email and determine the actions the recipient can take.
    Rules:
    - Return only actions that are explicitly supported by the email.
    - Choose only from:
        - Accept
        - Decline
        - Reschedule
        - Reply
        - Approve
        - Reject
        - None
    - If the recipient can choose to attend or not attend an event, interview, meeting, or invitation, return:
        Accept, Decline
    - If the email explicitly allows changing the date or time, also include:
        Reschedule
    - If the email requests approval, return:
        Approve, Reject
    - If the email only asks for a written response and no decision, return:
        Reply
    - If no action is required, return:
        None
    - Return only a comma-separated list of actions.
    - Do not include explanations, labels, or extra text.

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
                               max_new_tokens=20,
                               do_sample=False,
                               early_stopping=True)

    generate_token=output[0][inputs["input_ids"].shape[1]:]

    response=qwen1_tokenizer.decode(
        generate_token,
        skip_special_tokens=True
    ).strip()

    actions=[
        action.strip()
        for action in response.split(",")
    ]
    actions = [
        action for action in actions
        if action.lower() != "none"
    ]

    return actions


