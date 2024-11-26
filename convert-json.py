import json

# Function to convert train.src and train.tgt to valid.json format
def convert_to_json(src_file, tgt_file, json_file):
    translations = []

    with open(src_file, 'r', encoding='utf-8') as src, open(tgt_file, 'r', encoding='utf-8') as tgt:
        for src_line, tgt_line in zip(src, tgt):
            input_text = src_line.strip()
            output_text = tgt_line.strip()
            
            translations.append({
                "input": f"{input_text}",
                "output": output_text
            })

    # Write the translations to a JSON file
    with open(json_file, 'w', encoding='utf-8') as json_file:
        json.dump(translations, json_file, ensure_ascii=False, indent=4)

# Usage
convert_to_json('valid.src', 'valid.tgt', 'valid.json')