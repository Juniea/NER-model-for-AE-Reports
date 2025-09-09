#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import spacy
import html
import os
import random
import json

def split_sentences(nlp, text):
    """
    Splits the given text into sentences using spaCy's sentence segmenter.
    """
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def read_and_process_json_sections(json_file, nlp):
    global skipped_file
    """
    Reads a JSON file, processes sections into sentences, and maps mentions to sentences.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Process each page in the JSON data
    final_data = []
    for page_id, page_content in enumerate(data['pages']):
        print(f"\033[91mProcessing\033[0m Page ID: {page_id}")

        # Clean the content by replacing </p> with a space and removing <p> tags
        cleaned_content = re.sub(r'<\/?p>', '', page_content)
        cleaned_content = re.sub(r'[\r\n]+', '', cleaned_content)
        cleaned_content = re.sub(r'[\r\n\u21B5]+', ' ', page_content)

        # Extract entities from annotations
        entities = []
        for annotation in data['annotations']:
            if annotation['relationships']['date']['page']== page_id:
                start = annotation['textContext']['start']
                end = annotation['textContext']['end']
                entity_type = annotation['label'].lower()
                if entity_type in ["cdrug","sdrug"]:
                    entity_type = "drug"
                if entity_type in ["mae"]:
                    entity_type = "ae"
                mention_text = cleaned_content[start:end]
                # Append the entity tuple
                entities.append({
                    "start": start,
                    "end": end,
                    "type": entity_type,
                    "text": annotation['textContext']['text']
                })
                

        # Order entities based on the value of start
        entities = sorted(entities, key=lambda x: x['start'])
        print(f"Entities: {entities}")

        # Split the section into sentences
        sentences = split_sentences(nlp, cleaned_content)

        sentence_start = 0
        entity_index = 0  # Index to track the current entity being processed
        num_entities = len(entities)

        for sentence in sentences:
            sentence_end = sentence_start + len(sentence)
            sentence_entities = []

            # Process entities that belong to the current sentence
            while entity_index < num_entities and entities[entity_index]["start"] >= sentence_start and entities[entity_index]["end"] <= sentence_end:
                entity = entities[entity_index]
                # print(f"Entity: {entity}")
                # Calculate relative start and end positions in the sentence
                relative_start = entity["start"] - sentence_start
                relative_end = entity["end"] - sentence_start
                entity_text = entity["text"]
                start = sentence.find(entity_text)

                # Validate that the extracted text matches the entity text
                if start != -1:
                    end = start + len(entity_text)
                    entity_type = entity["type"]
                    global unique_entity_types
                    global entity_type_counts
                    unique_entity_types.add(entity_type)
                    if entity_type in entity_type_counts:
                        entity_type_counts[entity_type] += 1
                    else:
                        entity_type_counts[entity_type] = 1
                    print("sucessfully get the entity!!")
                    # if entity["type"] != "R/O" and entity["type"] != "sDx" and entity["type"] != "SYM" and entity["type"] != "Dx":
                    #     if entity["type"] in ["sDrug","cDrug", "drug"]:
                    #         sentence_entities.append((start, end, "Drug"))
                    #         print(f"\033[91mchanged to Drug: {entity['type']} at ({start}, {end}), and the word is {entity_text}\033[0m")
                    #     elif entity["type"] == "mAE" or entity["type"] == "ae":
                    #         sentence_entities.append((start, end, "AE"))
                    #         print(f"\033[91mchanged to AE: {entity['type']} at ({start}, {end}, and the word is {entity_text})\033[0m")
                    #     else:
                    #         sentence_entities.append((start, end, entity_type))
                    # else:
                    #     # print(f"\033[93mskip these entities: {entity['type']} at ({start}, {end})\033[0m")
                    #     skipped_file += 1
                        # print(f"skip these entities: {entity["type"]} at ({start}, {end})")
                    # sentence_entities.append((start, end, entity_type))
                    # sentence_entities.append((start, end, entity_type, entity_text))
                    # print(f"Entity matched: {entity_text} at ({start}, {end})")
                else:
                    print(
                        f"\033[93mWarning: Text mismatch for entity '{entity_text}' in {sentence} "
                        f"at indices ({relative_start}, {relative_end}).\033[0m" 
                    )

                # Move to the next entity
                entity_index += 1

            # Add the sentence and its entities to the labeled data
            if sentence_entities:
                final_data.append((sentence, {"entities": sentence_entities}))
                # print(f"\033[92mmatched Sentence: {sentence}\033[0m")
                # print(f"\033[92mmatched Entities: {sentence_entities}\033[0m")

            # Update the sentence_start for the next sentence
            if sentence_start + len(sentence) < len(cleaned_content) and cleaned_content[sentence_start + len(sentence)] == ' ':
                sentence_start = sentence_end + 1  # Account for the space/newline between sentences
            else:
                sentence_start = sentence_end

    return final_data

def get_file_list(folder_path):
    """
    Get a list of all JSON files in a folder.
    """
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".json")]


def split_file_list(file_list, train_ratio=0.8):
    """
    Split the list of files into train and test datasets.
    """
    random.seed(123)
    random.shuffle(file_list)
    split_idx = int(len(file_list) * train_ratio)
    return file_list[:split_idx], file_list[split_idx:]

def process_files(file_list, nlp):
    """
    Process JSON files into spaCy-compatible training data.
    """
    processed_data = []
    for json_file in file_list:
        print(f"\033[91mProcessing\033[0m file: {json_file}")
        processed_data.extend(read_and_process_json_sections(json_file, nlp))
    return processed_data
unique_entity_types = set()
entity_type_counts = {}
# Example usage
folder_path = 'C:/Users/Wenjuan.Zhang/OneDrive - FDA/Documents/Latest Annotation 07182025/FAERS_R1'  # Replace with your folder path
file_list = get_file_list(folder_path)

# Split into train and test datasets
train_files, test_files = split_file_list(file_list)

# Load a spaCy language model for sentence splitting
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")  # Add sentence segmenter
skipped_file = 0
# Process train and test files
train_data = process_files(train_files, nlp)
test_data = process_files(test_files, nlp)
print(test_files)
print(f"Skipped files: {skipped_file}")
# Output a sample from the formatted data
for sample in train_data[:3]:  # Display first 3 samples
    print(sample)
# Sort and print all unique entity types
sorted_unique_entity_types = sorted(unique_entity_types)
print(f"\033[92mUnique Entity Types: {sorted_unique_entity_types}\033[0m")
sorted_entity_type_counts = sorted(entity_type_counts.items())

# Print each key-value pair on a new line, separated by commas
for entity_type, count in sorted_entity_type_counts:
    print(f"{entity_type}: {count}")



# In[2]:


"This spontaneous report was received from a consumer or other non health professional (patient's family via a call center) and concerns a male male patient in his eighties from Japan: Local Case ID: 201412714.↵The patient's height and weight were not reported.↵The patient's concurrent conditions included back pain (underlying disease) and pain (underlying disease).↵The patient's history of drug abuse/illicit drug use, alcohol use and smoking was not reported.↵On an unspecified date, the patient started to receive treatment with glibenclamide (concomitant medication) at the department of internal medicine.↵On an unspecified date, the patient had low back pain recently and started treatment with TRAMCET (tramadol/apap, film coated tablet, oral) at the department of orthopedics.↵On an unspecified date, the patient had hypoglycaemia with clouding of consciousness many times.↵The physician told that the symptom was not a problem, but the blood glucose level frequently decreased to below 40 mg/dL.↵On an unspecified date, the dose of glibenclamide was reduced by half because the blood glucose level would increase to around 200 mg/dL if treatment with glibenclamide was discontinued.↵Action taken with tramadol/apap was not reported.↵The patient's outcome was not reported for hypoglycaemia.↵This report was serious (medically significant).↵Labeling Assessment and Seriousness:↵LLT hypoglycaemia (serious): CCDS listed, USPI unlisted, SmPC NA, IB NA↵Additional information was received via Grunenthal GmbH on 23-JUL-2014.↵Grunenthal GmbH reference number '2014-13564' was updated to the case."[815:871]


# In[3]:


# Filter the list to only keep strings that contain uppercase letters
filtered_entity_types = [entity for entity in unique_entity_types if any(char.isupper() for char in entity)]
entity_type_mapping = {entity.lower(): entity for entity in unique_entity_types}
# Print the filtered list
print(filtered_entity_types)
# List to be changed
to_change = ["ae", "drug"]

# Change the strings to the one with uppercase letters according to filtered_entity_types
changed_list = [entity for entity in filtered_entity_types if entity.lower() in to_change]
print(changed_list)


filtered_entity_types_lower = [entity.lower() for entity in filtered_entity_types]


# In[4]:


# removed_entity = ["SYM","ae-infectious","ae=infection<","ae\metabolism","bsym.sleeping","drug)lanuts<",
#                   "drug)methadone<","drug)morphine","drug)morphine<","lab=inflammatory","p=literature",
#                   "sDx"]

entities_with_low_count = [entity_type for entity_type, count in entity_type_counts.items() if count < 10]

# Print the list of entities with count < 10
print(entities_with_low_count)


# In[5]:


import re
import spacy
import html
import os
import random
import json

def split_sentences(nlp, text):
    """
    Splits the given text into sentences using spaCy's sentence segmenter.
    """
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def read_and_process_json_sections(json_file, nlp):
    global skipped_file
    """
    Reads a JSON file, processes sections into sentences, and maps mentions to sentences.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Process each page in the JSON data
    final_data = []
    for page_id, page_content in enumerate(data['pages']):
        print(f"\033[91mProcessing\033[0m Page ID: {page_id}")

        # Clean the content by replacing </p> with a space and removing <p> tags
        cleaned_content = re.sub(r'<\/?p>', '', page_content)
        cleaned_content = re.sub(r'[\r\n\u21B5]+', ' ', page_content)
        # Extract entities from annotations
        entities = []
        for annotation in data['annotations']:
            if annotation['relationships']['date']['page']== page_id:
                start = annotation['textContext']['start']
                end = annotation['textContext']['end']
                entity_type = annotation['label']
                mention_text = cleaned_content[start:end]
                # Append the entity tuple
                entities.append({
                    "start": start,
                    "end": end,
                    "type": entity_type.lower(),
                    "text": annotation['textContext']['text']
                })
                

        # Order entities based on the value of start
        entities = sorted(entities, key=lambda x: x['start'])
        print(f"Entities: {entities}")

        # Split the section into sentences
        sentences = split_sentences(nlp, cleaned_content)

        sentence_start = 0
        entity_index = 0  # Index to track the current entity being processed
        num_entities = len(entities)

        for sentence in sentences:
            sentence_end = sentence_start + len(sentence)
            sentence_entities = []

            # Process entities that belong to the current sentence
            while entity_index < num_entities and entities[entity_index]["start"] >= sentence_start and entities[entity_index]["end"] <= sentence_end:
                entity = entities[entity_index]
                # print(f"Entity: {entity}")
                # Calculate relative start and end positions in the sentence
                relative_start = entity["start"] - sentence_start
                relative_end = entity["end"] - sentence_start
                entity_text = entity["text"]
                start = sentence.find(entity_text)

                # Validate that the extracted text matches the entity text
                if start != -1:
                    end = start + len(entity_text)
                    entity_type = entity["type"]
                    global unique_entity_types
                    global entity_type_counts
                    unique_entity_types.add(entity_type)
                    if entity_type in entity_type_counts:
                        entity_type_counts[entity_type] += 1
                    else:
                        entity_type_counts[entity_type] = 1
                    # if entity["type"] != "R/O" and entity["type"] != "sDx" and entity["type"] != "SYM" and entity["type"] != "Dx":
                    #     if entity["type"] in ["sDrug","cDrug", "drug"]:
                    #         sentence_entities.append((start, end, "Drug"))
                    #         print(f"\033[91mchanged to Drug: {entity['type']} at ({start}, {end}), and the word is {entity_text}\033[0m")
                    #     elif entity["type"] == "mAE" or entity["type"] == "ae":
                    #         sentence_entities.append((start, end, "AE"))
                    #         print(f"\033[91mchanged to AE: {entity['type']} at ({start}, {end}, and the word is {entity_text})\033[0m")
                    #     else:
                    #         sentence_entities.append((start, end, entity_type))
                    # else:
                    #     # print(f"\033[93mskip these entities: {entity['type']} at ({start}, {end})\033[0m")
                    #     skipped_file += 1
                        # print(f"skip these entities: {entity["type"]} at ({start}, {end})")
                    # sentence_entities.append((start, end, entity_type))
                    # sentence_entities.append((start, end, entity_type, entity_text))
                    # print(f"Entity matched: {entity_text} at ({start}, {end})")
                else:
                    print(
                        f"\033[93mWarning: Text mismatch for entity '{entity_text}' in {sentence} "
                        f"at indices ({relative_start}, {relative_end}).\033[0m" 
                    )

                # Move to the next entity
                entity_index += 1

            # Add the sentence and its entities to the labeled data
            if sentence_entities:
                final_data.append((sentence, {"entities": sentence_entities}))
                # print(f"\033[92mmatched Sentence: {sentence}\033[0m")
                # print(f"\033[92mmatched Entities: {sentence_entities}\033[0m")

            # Update the sentence_start for the next sentence
            if sentence_start + len(sentence) < len(cleaned_content) and cleaned_content[sentence_start + len(sentence)] == ' ':
                sentence_start = sentence_end + 1  # Account for the space/newline between sentences
            else:
                sentence_start = sentence_end

    return final_data

def get_file_list(folder_path):
    """
    Get a list of all JSON files in a folder.
    """
    return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".json")]


def split_file_list(file_list, train_ratio=0.8):
    """
    Split the list of files into train and test datasets.
    """
    random.seed(123)
    random.shuffle(file_list)
    split_idx = int(len(file_list) * train_ratio)
    return file_list[:split_idx], file_list[split_idx:]

def process_files(file_list, nlp):
    """
    Process JSON files into spaCy-compatible training data.
    """
    processed_data = []
    for json_file in file_list:
        print(f"\033[91mProcessing\033[0m file: {json_file}")
        processed_data.extend(read_and_process_json_sections(json_file, nlp))
    return processed_data
unique_entity_types = set()
entity_type_counts = {}
# Example usage
folder_path = 'C:/Users/Wenjuan.Zhang/OneDrive - FDA/Documents/Latest Annotation 07182025/FAERS_R1'  # Replace with your folder path
file_list = get_file_list(folder_path)

# Split into train and test datasets
train_files, test_files = split_file_list(file_list)

# Load a spaCy language model for sentence splitting
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")  # Add sentence segmenter
skipped_file = 0
# Process train and test files
train_data = process_files(train_files, nlp)
test_data = process_files(test_files, nlp)
print(test_files)
print(f"Skipped files: {skipped_file}")
# Output a sample from the formatted data
for sample in train_data[:3]:  # Display first 3 samples
    print(sample)
# Sort and print all unique entity types
sorted_unique_entity_types = sorted(unique_entity_types)
print(f"\033[92mUnique Entity Types: {sorted_unique_entity_types}\033[0m")
sorted_entity_type_counts = sorted(entity_type_counts.items())

# Print each key-value pair on a new line, separated by commas
for entity_type, count in sorted_entity_type_counts:
    print(f"{entity_type}: {count}")


# %%
import re
import spacy
import html
import os
import random
import json

def split_sentences(nlp, text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def read_and_process_json_sections(json_file, nlp, file_type):
    global skipped_file
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    final_data = []

    for page_id, page_content in enumerate(data['pages']):
        # Replace <p>, </p> and \n with spaces
        cleaned_content = re.sub(r'</?p>', ' ', page_content)
        cleaned_content = re.sub(r'\n', ' ', cleaned_content)
        cleaned_content = re.sub(r'[\r\n\u21B5]+', ' ', page_content)

        # Extract and filter entities
        entities = []
        for annotation in data['annotations']:
            if annotation['relationships']['date']['page'] != page_id:
                continue
            if annotation['note'] not in file_type:
                continue

            start = annotation['textContext']['start']
            end = annotation['textContext']['end']
            entity_type = annotation['label'].lower()
            mention_text = cleaned_content[start:end]

            # if entity_type in ["drug", "cdrug", "sdrug"]:
            #     entity_type = "drug"
            # elif entity_type == "mae":
            #     entity_type = "ae"
            # elif entity_type in ["fhx", "mhx"]:
            #     entity_type = "hx"

            if entity_type not in ["fhx","ro","time","latency"]:
                entities.append({
                    "start": start,
                    "end": end,
                    "type": entity_type,
                    "text": annotation['textContext']['text']
                })

        # Sort entities by starting position
        entities.sort(key=lambda x: x['start'])

        sentences = split_sentences(nlp, cleaned_content)
        sentence_start = 0
        entity_index = 0
        num_entities = len(entities)

        for sentence in sentences:
            sentence_end = sentence_start + len(sentence)
            sentence_entities = []

            while (entity_index < num_entities and
                   sentence_start <= entities[entity_index]["start"] < sentence_end and
                   sentence_start <= entities[entity_index]["end"] <= sentence_end):

                entity = entities[entity_index]
                entity_text = entity["text"]
                relative_start = entity["start"] - sentence_start
                relative_end = entity["end"] - sentence_start

                # Find the actual position of entity text in the sentence
                match_start = sentence.find(entity_text)
                if match_start != -1:
                    match_end = match_start + len(entity_text)
                    ent_type = entity["type"]

                    # Standardize case and match entity types
                    if ent_type.islower() and ent_type in filtered_entity_types_lower:
                        matched_type = [t for t in filtered_entity_types if t.lower() == ent_type][0]
                        print(f"\033[91mChanged to {matched_type}: {ent_type} at ({match_start}, {match_end}) with word '{entity_text}'\033[0m")
                        ent_type = matched_type

                    if ent_type not in entities_with_low_count:
                        unique_entity_types.add(ent_type)
                        entity_type_counts.setdefault(ent_type, 0)
                        entity_type_counts[ent_type] += 1

                        sentence_entities.append((
                            match_start, match_end, ent_type
                        ))
                    else:
                        print(f"\033[91mLow-count entity: {entity['type']} at ({match_start}, {match_end})\033[0m")
                else:
                    print(f"\033[93mWarning: Text mismatch for entity '{entity_text}' in sentence: \"{sentence}\" at ({relative_start}, {relative_end})\033[0m")

                entity_index += 1

            if sentence_entities:
                final_data.append((sentence, {"entities": sentence_entities}))

            # Update sentence start position for next iteration
            if sentence_start + len(sentence) < len(cleaned_content) and cleaned_content[sentence_start + len(sentence)] == ' ':
                sentence_start = sentence_end + 1
            else:
                sentence_start = sentence_end

    return final_data
def process_files(file_list, nlp, file_type):
    """
    Process JSON files into spaCy-compatible training data.
    """
    processed_data = []
    for json_file in file_list:
        # print(f"\033[91mProcessing\033[0m file: {json_file}")
        processed_data.extend(read_and_process_json_sections(json_file, nlp, file_type))
    return processed_data


# %%
print(len(train_files))
print(len(test_files))

# %%
###train data with SME1 and test data with SME1, too
train_data = process_files(train_files, nlp, ["LLM"])  #"LLM","SME1"
test_data = process_files(test_files, nlp, ["LLM"])
total_data = train_data + test_data
print(test_files)
print(f"Skipped files: {skipped_file}")
# Output a sample from the formatted data
for sample in train_data[:3]:  # Display first 3 samples
    print(sample)

# %%
print(entity_type_counts)

# %%
# import csv


# # 定义CSV的列名
# fieldnames = ["start", "end", "label", "text", "sentence", "origin_start", "origin_end", "origin_type", "origin_label", "json_file"]

# with open("C:/Users/Wenjuan.Zhang/OneDrive - FDA/Documents/total_data.csv", mode="w", newline="", encoding="utf-8") as file:
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()  # 写入列名

#     for text, info in total_data:
#         for entity in info['entities']:
#             row = {
#                 "start": entity[0],
#                 "end": entity[1],
#                 "label": entity[2],
#                 "text": entity[3],
#                 "sentence":entity[4],
#                 "origin_start":entity[5],
#                 "origin_end":entity[6],
#                 "origin_type":entity[7],
#                 "origin_label":entity[8],
#                 "json_file": entity[9]
#             }
#             writer.writerow(row)  #


# %%

skipped_count = 0
remained_count = 0
exceedlength = 0
import warnings
from pathlib import Path
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
import re
from collections import defaultdict

def validate_entities(entities):
    """
    Validate that there are no overlapping entities.
    Args:
        entities (list): List of tuples (start, end, label).
    Returns:
        bool: True if no overlaps, False otherwise.
    """
    entities = sorted(entities, key=lambda x: x[0])  # Sort by start index
    for i in range(len(entities) - 1):
        if entities[i][1] > entities[i + 1][0]:  # Check for overlap
            return False
    return True

def clean_entities(entities, text):
    """
    Clean entity spans by removing leading and trailing whitespace.
    Args:
        entities (list): List of tuples (start, end, label).
        text (str): The text containing the entities.
    Returns:
        list: Cleaned list of entities.
    """
    cleaned_entities = []
    for start, end, label in entities:
        # Ensure start and end are within bounds
        start = max(0, min(start, len(text)))
        end = max(0, min(end, len(text)))
        
        # Remove leading whitespace
        while start < end and text[start].isspace():
            start += 1
        # Remove trailing whitespace
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            cleaned_entities.append((start, end, label))
    return cleaned_entities

def split_sentences(nlp, text, entities):
    """
    Splits the given text into sentences using spaCy's sentence segmenter.
    If a sentence is longer than 512 characters, it is split at the closest whitespace to the left of the 512th character.
    Adjusts entity positions accordingly.
    """
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    split_sentences = []
    split_entities = []

    for sentence in sentences:
        while len(sentence) > 512:
            # Find the closest whitespace to the left of the 512th character
            split_index = sentence.rfind(' ', 0, 512)
            if split_index == -1:
                split_index = 512  # If no whitespace is found, split at 512

            # Split the sentence at the whitespace
            part1 = sentence[:split_index].strip()
            part2 = sentence[split_index:].strip()
            split_sentences.append(part1)

            # Adjust entity positions
            part1_len = len(part1)
            part1_entities = []
            part2_entities = []
            for start, end, label in entities:
                if start < part1_len:
                    if end <= part1_len:
                        part1_entities.append((start, end, label))
                    else:
                        part1_entities.append((start, part1_len, label))
                        part2_entities.append((0, end - part1_len, label))
                else:
                    part2_entities.append((start - part1_len, end - part1_len, label))
            split_entities.append(part1_entities)
            sentence = part2
            entities = part2_entities

        split_sentences.append(sentence)
        split_entities.append(entities)

    return split_sentences, split_entities

def truncate_text_and_entities(text, entities, max_length=512):
    """
    Truncate text and entities to a maximum length.
    Args:
        text (str): The text to be truncated.
        entities (list): List of tuples (start, end, label).
        max_length (int): Maximum length of the text.
    Returns:
        tuple: Truncated text and entities.
    """
    global exceedlength
    if len(text) <= max_length:
        return text, entities

    truncated_text = text[:max_length]
    truncated_entities = []
    for start, end, label in entities:
        if start < max_length:
            if end > max_length:
                exceedlength += 1
                end = max_length
            truncated_entities.append((start, end, label))
    return truncated_text, truncated_entities

def convert(lang: str, TRAIN_DATA, output_path: Path):
    """
    Convert training data into SpaCy's .spacy format.
    Args:
        lang (str): Language code (e.g., "en").
        TRAIN_DATA (list): List of training data in the format [(text, {"entities": [...]})].
        output_path (Path): Path to save the .spacy file.
    """
    global skipped_count
    global remained_count
    nlp = spacy.blank(lang)  # Create a blank language model
    nlp.add_pipe("sentencizer")  # Add the sentencizer component to the pipeline
    db = DocBin()  # Initialize DocBin to store Doc objects
    entity_counts = defaultdict(int)
    for text, annot in TRAIN_DATA:
        # Split sentences and adjust entities
        sentences, entities_list = split_sentences(nlp, text, annot["entities"])
        for sentence, entities in zip(sentences, entities_list):
            # Truncate text and entities to the maximum length
            sentence, entities = truncate_text_and_entities(sentence, entities, max_length=512)
            doc = nlp.make_doc(sentence)  # Create a new Doc object
            
            # Validate entities for overlaps
            if not validate_entities(entities):
                skipped_count += 1
                warnings.warn(f"Skipping text with overlapping entities: {repr(sentence)}")
                continue
            
            # Clean entity spans
            entities = clean_entities(entities, sentence)
            
            ents = []
            for start, end, label in entities:
                span = doc.char_span(start, end, label=label, alignment_mode="strict")  #strict,expand
                if span is None:
                    skipped_count += 1
                    msg = (f"Skipping entity [{start}, {end}, {label}] in the text because the character span "
                           f"'{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(sentence)}\n")
                    warnings.warn(msg)
                else:
                    remained_count += 1
                    print("Successfully added entity")
                    ents.append(span)
                    entity_counts[label] += 1
            filtered = filter_spans(ents) # THIS DOES THE TRICK
            if filtered:
                doc.ents = filtered  # Set the entities for the Doc
                db.add(doc)  # Add the Doc to DocBin

    db.to_disk(output_path)  # Save the DocBin to disk
    print(f"Saved to {output_path}")
    return entity_counts

# Example Usage

# Output paths
train_output_path = Path("C:/Users/Wenjuan.Zhang/OneDrive - FDA/Documents/train_FAVERS_R1_v1_LLM_new_revision.spacy")
test_output_path = Path("C:/Users/Wenjuan.Zhang/OneDrive - FDA/Documents/valid_FAVERS_R1_v1_LLM_new_revision.spacy")
total_output_path = Path("C:/Users/Wenjuan.Zhang/OneDrive - FDA/Documents/total_FAVERS_R1_v1_LLM_new_revision.spacy")
# Convert and save to .spacy format
train_entity_counts = convert("en", train_data, train_output_path)
test_entity_counts = convert("en", test_data, test_output_path)
convert("en", total_data, total_output_path)

# %%

train_data


# %%
# Print the counts
print("Train Entity Counts:")
for entity_type, count in train_entity_counts.items():
    print(f"{entity_type}: {count}")

print("\nTest Entity Counts:")
for entity_type, count in test_entity_counts.items():
    print(f"{entity_type}: {count}")




# In[6]:


print(len(train_files))
print(len(test_files))


# In[7]:


print(train_data[14])
print(test_data[0])
print(len(train_data))
print(len(test_data))
print("The patient's heart rate changed from 240/min to 100/min."[14:56])

