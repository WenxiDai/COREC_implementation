import json
import csv
import gc
import re
from tqdm import tqdm
import html  # Needed for HTML unescaping

def clean_text(text):
    # Ensure text is a string (handle list or other types)
    if isinstance(text, list):
        text = ' '.join(str(t) for t in text)
    elif not isinstance(text, str):
        text = str(text)

    # Decode HTML entities, e.g., &amp; -> &
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Remove control characters (including \x00-\x1F and \x7F)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)

    # Replace common escape characters
    text = text.replace('\\"', '"').replace('\\\'', "'").replace('\\\\', '\\')

    # Remove brackets, braces, and redundant quotes
    text = re.sub(r'[\[\]{}"]+', '', text)

    # Remove non-printable unicode characters like \u2028, \u2029
    text = re.sub(r'[\u2028\u2029]', '', text)

    # Finally, clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_rank_number(rank_raw):
    """
    Extract the rank number (as integer) from the 'rank' field.
    Return None if parsing fails or input is invalid.
    """
    if not isinstance(rank_raw, list) or len(rank_raw) == 0:
        return None

    raw = rank_raw[0]
    match = re.search(r"#(\d+)", raw)
    if match:
        return int(match.group(1))
    return None

def process_meta_in_chunks(input_path, output_path, chunk_size=100000):
    required_columns = ['asin', 'price', 'description', 'brand', 'rank_number', 'title']
    chunk = []
    
    with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(required_columns)  # Write CSV header row

        for line in tqdm(infile, desc="Processing metadata in chunks"):
            if line.strip():
                chunk.append(line)

            # Process chunk if it reaches the defined size
            if len(chunk) >= chunk_size:
                process_chunk(chunk, writer, required_columns)
                chunk = []
                gc.collect()
        
        # Process the last remaining chunk
        if chunk:
            process_chunk(chunk, writer, required_columns)

    print(f"Finished writing filtered metadata to {output_path}")

def process_chunk(chunk, writer, required_columns):
    for line in chunk:
        try:
            record = json.loads(line)
            asin = record.get('asin')
            if not asin:
                continue

            filtered_row = []
            valid = False
            for col in required_columns:
                val = record.get(col, '')
                if isinstance(val, list):
                    val = json.dumps(val)
                if col == 'rank_number':
                    val = extract_rank_number(val)
                filtered_row.append(val)
                if col != 'asin' and val:
                    valid = True  # At least one field other than asin must be non-empty
            
            if valid:
                writer.writerow(filtered_row)
        except json.JSONDecodeError:
            continue

if __name__ == "__main__":
    input_file = 'meta_Home_and_Kitchen.json'
    output_file = 'meta_Home_and_Kitchen_filtered.csv'
    process_meta_in_chunks(input_file, output_file)