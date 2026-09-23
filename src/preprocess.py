import re
from pathlib import Path
import html


def load_email(email):
    with open (email, "r" , encoding="utf-8") as file:
        email = file.read()
    return email

def remove_html(email):
    if not email:
        return ""
    email=re.sub(r'<(br|p|div)[^>]*>','\n',email,flags=re.IGNORECASE)
    email=re.sub(r'<[^>]+>','',email)
    email=html.unescape(email)
    email=re.sub(r'\n\s*\n+','\n\n',email)
    return email.strip()
def clean_email(email):
    email=email.strip()
    email=re.sub(r'\s+',' ',email)
    return email


def preprocessing(email):
    #email=load_email(file_path)
    email=remove_html(email)
    email=clean_email(email)
    return email

