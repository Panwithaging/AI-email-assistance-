from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

import torch

from preprocess import preprocessing

from models import qwen3_tokenizer,qwen3_model

def reply(text,action,preferred_date=None,preferred_time=None):
    prompt=None

    accept_prompt=f"""You are an AI email assistant.
    Write a professional email reply.
    
    Rules:
    - Reply as the recipient of the email.
    - The recipient has decided to ACCEPT.
    - Confirm attendance or acceptance.
    - Be polite, concise, and professional.
    - Do not invent facts, dates, names, or commitments.
    - Do not change any information given in the email.
    - Begin with "Dear Sir/Madam,".
    - End with:
        Best regards,
        [Your Name]
    - Return only the reply.
    
    Email:
    {text}"""

    decline_prompt=f"""You are an AI email assistant.
    Write a professional email reply.
    
    Rules:
    - Reply as the recipient of the email.
    - The recipient has decided to DECLINE.
    - Politely decline the invitation or request.
    - Thank the sender when appropriate.
    - Do not invent reasons unless they are provided.
    - Be polite, concise, and professional.
    - Begin with "Dear Sir/Madam,".
    - End with:
        Best regards,
        [Your Name]
    - Return only the reply.
    
    Email:
    {text}"""

    reschedule_prompt=f"""You are an AI email assistant.
    Write a professional email reply.
    
    Rules:
    - Reply as the recipient of the email.
    - The recipient has decided to RESCHEDULE.
    - Politely explain that you are unavailable at the scheduled date or time.
    - Request to reschedule the interview or meeting.
    - Ask whether <DATE> at <TIME> would work for the sender.
    - Use the placeholders <DATE> and <TIME> exactly as written.
    - Do not replace, modify, calculate, infer, or invent another date or time.
    - Do not say "convenient for me".
    - Be polite, concise, and professional.
    - Begin with "Dear Sir/Madam,".
    - End with:
        Best regards,
        [Your Name]
    - Return only the reply.
    
    Email:
    {text}"""

    normal_reply=f"""You are an AI email assistant.
    Write a professional email reply.
    Rules:
    - Reply as the recipient of the email.
    - Directly address the sender's request or question.
    - If the email requests confirmation, provide confirmation only if appropriate.
    - If the email asks a question that cannot be answered from the email, politely state that additional information is needed.
    - Do not invent facts, dates, names, promises, or commitments.
    - Do not repeat the original email.
    - Keep the reply concise and professional.
    - Begin with "Dear Sir/Madam,".
    - End with:
        Best regards,
        [Your Name]
    - Return only the reply.
    
    Email:
    {text}"""

    if action =="Accept":
        prompt=accept_prompt
    elif action == "Decline":
        prompt =decline_prompt
    elif action =="Reschedule":
        prompt=reschedule_prompt
    else:
        prompt=normal_reply
    message=[
        {
            "role":"user",
            "content":prompt
        }
    ]

    formated_prompt=qwen3_tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs=qwen3_tokenizer(
        formated_prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        output=qwen3_model.generate(inputs["input_ids"],
                              attention_mask=inputs["attention_mask"],
                              max_new_tokens=150,
                              num_beams=7,
                              early_stopping=True)
    generate_tokens=output[0][inputs["input_ids"].shape[1]:]
    email_reply=qwen3_tokenizer.decode(
        generate_tokens,
        skip_special_tokens=True
    ).strip()
    if preferred_date:
        email_reply=email_reply.replace(
                "<DATE>",
                preferred_date
            )
    if preferred_time:
        email_reply=email_reply.replace(
                "<TIME>",
                preferred_time
            )

    return email_reply

    


