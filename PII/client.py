from typing import List, Optional, Tuple, Dict
from pii import TextAnalyzerService
import pandas as pd

if __name__ == "__main__":
    # Load the CSV file
    # df = pd.read_csv(r"C:\Users\INDIA AI DATA LABS\Downloads\INTEL\Project X\reviews.csv") #Absolute Path
    df = pd.read_csv(r"./reviews.csv") #Relative path

    # Create an instance of TextAnalyzerService with the desired model
    text_analyzer_service_model1 = TextAnalyzerService(model_choice="obi/deid_roberta_i2b2")

    # Define lists to store anonymized and deanonymized texts
    anonymized_texts = []
    deanonymized_texts = []

    # Iterate over each row in the first column of the DataFrame
    for index, row in df.iterrows():
        text = row[0]  # Assuming the first column is the one you want to anonymize

        # Analyze text with the chosen model
        entities_model1 = text_analyzer_service_model1.analyze_text(text)

        # Anonymize text
        anonymized_text, req_dict = text_analyzer_service_model1.anonymize_text(text, entities_model1, operator="encrypt")

        # Deanonymize text
        deanonymized_text = text_analyzer_service_model1.deanonymize_text(anonymized_text)

        # Append anonymized and deanonymized texts to lists
        anonymized_texts.append(anonymized_text.text)
        deanonymized_texts.append(deanonymized_text.text)

    # Add anonymized and deanonymized texts as new columns in the DataFrame
    df['Anonymized_Text'] = anonymized_texts
    df['Deanonymized_Text'] = deanonymized_texts

    # Drop "Deanonymized_Text" and "Rate" columns
    #df = df.drop(columns=["Deanonymized_Text","rating"])   #Commented

    # Save the DataFrame to a new CSV file
    df.to_csv("output_anonymized.csv", index=False)

