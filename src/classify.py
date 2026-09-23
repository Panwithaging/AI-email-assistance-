from preprocess import preprocessing

from transformers import pipeline

from models import classifier


labels = [
        "Recruitment",
        "Meeting",
        "Finance",
        "Office",
        "Personal",
        "Spam",
        "Invitation"
        ]
def classify(text):
    result=classifier(text,labels)

    categorical=result["labels"][0]

    return categorical

