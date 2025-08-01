import pandas as pd
import json
from tqdm import tqdm
import os
import gc
import numpy as np
from collections import defaultdict, Counter
import re, html

def clean_text(text):
    # Ensure text is a string (handle list or other types)
    if isinstance(text, list):
        text = ' '.join(str(t) for t in text)
    elif not isinstance(text, str):
        text = str(text)
    # Decode HTML entities (e.g. &amp; -> &)
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)

    # Remove control characters (\x00-\x1F and \x7F)
    text = re.sub(r'[\x00-\x1F\x7F]', '', text)

    # Replace common escape characters
    text = text.replace('\\"', '"').replace('\\\'', "'").replace('\\\\', '\\')

    # Remove brackets, braces, and redundant quotes
    text = re.sub(r'[\[\]{}"]+', '', text)

    # Remove unprintable Unicode characters (e.g. \u2028, \u2029)
    text = re.sub(r'[\u2028\u2029]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def load_meta_data_in_chunks(path, chunk_size=100000):
    """Load metadata in chunks to reduce memory usage"""
    meta_dict = {}
    required_columns = ['asin', 'title', 'price', 'description', 'brand', 'rank']
    chunk_count = 0
    temp_count = 0
    
    with open(path, 'r') as f:
        chunk = []
        for line in tqdm(f, desc="Loading metadata"):
            temp_count += 1
            try:
                if len(line.strip()) > 0:
                    chunk.append(line)
                
                # Process chunk once size limit is reached
                if len(chunk) >= chunk_size:
                    process_meta_chunk(chunk, meta_dict, required_columns)
                    chunk = []
                    chunk_count += 1
                    gc.collect()
                    print(f"Processed {chunk_count * chunk_size + len(chunk)} metadata records")
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
        
        # Process the last chunk
        if chunk:
            process_meta_chunk(chunk, meta_dict, required_columns)
    
    print(f"Loaded {len(meta_dict)} product metadata entries")
    return meta_dict

def process_meta_chunk(chunk, meta_dict, required_columns):
    """Process one chunk of metadata"""
    for line in chunk:
        try:
            record = json.loads(line)
            asin = record.get('asin')

            if not asin:
                continue
        
            price = record.get('price', '')
            if not (isinstance(price, str) and re.match(r'^\$\d+(?:\.\d{2})?$', price)):
                continue

            title = record.get('title', '')
            description = record.get('description')
            brand = record.get('brand')
            rank_raw = record.get('rank')

            # Skip if any required field is missing
            if not (title and description and brand and rank_raw):
                continue

            # Extract minimum rank from list
            min_rank = None
            rank_entries = rank_raw if isinstance(rank_raw, list) else [rank_raw]
            for r in rank_entries:
                match = re.search(r'#([\d,]+)', r)
                if match:
                    rank_val = int(match.group(1).replace(',', ''))
                    if min_rank is None or rank_val < min_rank:
                        min_rank = rank_val
            if min_rank is None:
                continue
            rank_str = f"#{min_rank:,}"

            filtered_record = {
                'asin': asin,
                'title': title,
                'price': price,
                'description': description,
                'brand': brand,
                'rank': rank_str
            }

            meta_dict[asin] = filtered_record
        except json.JSONDecodeError:
            continue

def find_active_users(reviews_path, meta_dict, min_reviews=30, max_users=None):
    """Find active users with at least min_reviews interactions and select top max_users"""
    user_reviews = defaultdict(list)
    
    # Count interactions for each user
    with open(reviews_path, 'r') as f:
        for line in tqdm(f, desc="Collecting user interactions"):
            try:
                review = json.loads(line)
                # Check required fields and whether the item is in metadata
                if all(field in review for field in ['reviewerID', 'asin', 'unixReviewTime']) and review['asin'] in meta_dict:
                    user_reviews[review['reviewerID']].append({
                        'asin': review['asin'],
                        'overall': review.get('overall', 0),
                        'unixReviewTime': review.get('unixReviewTime', 0)
                    })
            except json.JSONDecodeError:
                continue
    
    # Filter users with at least min_reviews interactions
    qualified_users = {}
    for user_id, reviews in user_reviews.items():
        if len(reviews) >= min_reviews:
            qualified_users[user_id] = len(reviews)
    
    print(f"Found {len(qualified_users)} users with at least {min_reviews} interactions")

    # Global stats
    total_qualified_users = len(qualified_users)
    total_interactions_qualified = 0
    qualified_item_set = set()
    for uid in qualified_users:
        reviews_list = user_reviews[uid]
        total_interactions_qualified += len(reviews_list)
        qualified_item_set.update(r['asin'] for r in reviews_list)

    total_qualified_items = len(qualified_item_set)
    density_all = (total_interactions_qualified /
                   (total_qualified_users * total_qualified_items)) if total_qualified_items else 0
    avg_seq_len_all = (total_interactions_qualified / total_qualified_users) if total_qualified_users else 0

    global_stats = {
        'total_users': total_qualified_users,
        'total_items': total_qualified_items,
        'total_interactions': total_interactions_qualified,
        'density': density_all,
        'avg_seq_length': avg_seq_len_all
    }

    # Select top users if limit is set
    if max_users and max_users > 0:
        sorted_users = sorted(qualified_users.items(), key=lambda x: x[1], reverse=True)
        selected_users = {u: user_reviews[u] for u, _ in sorted_users[:max_users]}
        print(f"Selected top {len(selected_users)} users")
    else:
        selected_users = {u: user_reviews[u] for u in qualified_users}
        print(f"Selected all {len(selected_users)} users")

    del user_reviews
    gc.collect()

    return selected_users, global_stats

def process_user_sequences(user_reviews, meta_dict, min_records=30, max_records=50, output_dir='sequential_data'):
    """Process user reviews and split into train/val/test using recent records"""
    os.makedirs(output_dir, exist_ok=True)
    
    train_data, val_data, test_data = [], [], []
    train_count, val_count, test_count = 0, 0, 0
    
    for user_id, reviews in tqdm(user_reviews.items(), desc="Processing user sequences"):
        reviews.sort(key=lambda x: x['unixReviewTime'], reverse=True)
        recent_reviews = reviews[:max_records]

        if len(recent_reviews) >= min_records:
            recent_reviews.sort(key=lambda x: x['unixReviewTime'])

            num_records = len(recent_reviews)

            # Ensure at least 1 record per split
            if num_records == 3:
                train_size, val_size = 1, 1
            elif num_records == 4:
                train_size, val_size = 2, 1
            else:
                train_size = max(1, int(num_records * 0.8))
                val_size = max(1, int(num_records * 0.1))
                if train_size + val_size >= num_records:
                    train_size = num_records - 2
                    val_size = 1
            
            train_reviews = recent_reviews[:train_size]
            val_reviews = recent_reviews[train_size:train_size + val_size]
            test_reviews = recent_reviews[train_size + val_size:]

            for review in test_reviews:
                record = {'reviewerID': user_id, **review}
                meta = meta_dict.get(record['asin'], {})
                record.update({
                    'price': meta.get('price', ''),
                    'description': clean_text(meta.get('description', '')),
                    'title': clean_text(meta.get('title', '')),
                    'brand': clean_text(meta.get('brand', '')),
                    'rank': clean_text(meta.get('rank', ''))
                })
                test_data.append(record)
                test_count += 1

            for review in val_reviews:
                record = {'reviewerID': user_id, **review}
                meta = meta_dict.get(record['asin'], {})
                record.update({
                    'price': meta.get('price', ''),
                    'description': clean_text(meta.get('description', '')),
                    'title': clean_text(meta.get('title', '')),
                    'brand': clean_text(meta.get('brand', '')),
                    'rank': clean_text(meta.get('rank', ''))
                })
                val_data.append(record)
                val_count += 1

            for review in train_reviews:
                record = {'reviewerID': user_id, **review}
                meta = meta_dict.get(record['asin'], {})
                record.update({
                    'price': meta.get('price', ''),
                    'title': clean_text(meta.get('title', '')),
                    'description': clean_text(meta.get('description', '')),
                    'brand': clean_text(meta.get('brand', '')),
                    'rank': clean_text(meta.get('rank', ''))
                })
                train_data.append(record)
                train_count += 1

    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, 'valid.json'), 'w') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print("Data split complete!")
    print(f"Training set: {train_count} records")
    print(f"Validation set: {val_count} records")
    print(f"Test set: {test_count} records")
    
    return train_count, val_count, test_count

def create_sasrec_mapping(train_file, val_file, test_file, output_dir):
    """Create user/item ID mappings for SASRec"""
    all_users = set()
    all_items = set()
    
    for filepath in [train_file, val_file, test_file]:
        with open(filepath, 'r') as f:
            reader = json.load(f)
            for item in tqdm(reader, desc=f"Mapping {os.path.basename(filepath)}"):
                if 'reviewerID' in item and 'asin' in item:
                    all_users.add(item['reviewerID'])
                    all_items.add(item['asin'])
    
    user_map = {user: i for i, user in enumerate(all_users)}
    item_map = {item: i + 1 for i, item in enumerate(all_items)}  # 0 reserved for padding

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'user_map.json'), 'w') as f:
        json.dump(user_map, f)
    with open(os.path.join(output_dir, 'item_map.json'), 'w') as f:
        json.dump(item_map, f)

    return user_map, item_map

def create_sasrec_sequences(train_file, val_file, test_file, user_map, item_map, output_dir):
    """Generate user-item interaction sequences in SASRec format"""
    user_sequences = defaultdict(list)
    
    for filepath in [train_file, val_file, test_file]:
        with open(filepath, 'r') as f:
            reader = json.load(f)
            for row in tqdm(reader, desc=f"Processing {os.path.basename(filepath)}"):
                if 'reviewerID' in row and 'asin' in row and 'unixReviewTime' in row:
                    user_id = user_map[row['reviewerID']]
                    item_id = item_map[row['asin']]
                    timestamp = int(row['unixReviewTime'])
                    user_sequences[user_id].append((item_id, timestamp))
    
    with open(os.path.join(output_dir, 'sequential_data.txt'), 'w') as f:
        for user_id, interactions in tqdm(user_sequences.items(), desc="Writing sequences"):
            sorted_interactions = sorted(interactions, key=lambda x: x[1])
            for item_id, _ in sorted_interactions:
                f.write(f"{user_id} {item_id}\n")

def improved_preprocessing(reviews_path, meta_path, output_dir='dataset/sequential/home_and_kitchen', 
                          sasrec_dir='sasrec_home_and_kitchen', min_reviews=30, max_users=3000,
                          min_records=30, max_records=50):
    """Main function for improved preprocessing with user selection and sequence generation"""
    print(f"Starting preprocessing, targeting {max_users} users with {min_reviews}+ reviews, keeping {min_records}-{max_records} recent records")
    
    print("Step 1: Loading metadata...")
    meta_dict = load_meta_data_in_chunks(meta_path)

    print("Step 2: Finding active users...")
    selected_user_reviews, global_stats = find_active_users(reviews_path, meta_dict, min_reviews, max_users)
    
    print("Step 3: Processing user sequences and splitting datasets...")
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, 'train.json')
    val_file = os.path.join(output_dir, 'valid.json')
    test_file = os.path.join(output_dir, 'test.json')

    train_count, val_count, test_count = process_user_sequences(
        selected_user_reviews, meta_dict, min_records, max_records, output_dir)

    del meta_dict
    gc.collect()

    # Global stats before user limitation
    print("\n--- Statistics for all qualified users (before limiting to top users) ---")
    print(f"Total users: {global_stats['total_users']}")
    print(f"Total items: {global_stats['total_items']}")
    print(f"Total interactions: {global_stats['total_interactions']}")
    print(f"Density: {global_stats['density']:.6f}")
    print(f"Average sequence length: {global_stats['avg_seq_length']:.2f}")

    print("Step 4: Creating SASRec mappings...")
    os.makedirs(sasrec_dir, exist_ok=True)
    user_map, item_map = create_sasrec_mapping(train_file, val_file, test_file, sasrec_dir)

    print("Step 5: Creating SASRec sequences...")
    create_sasrec_sequences(train_file, val_file, test_file, user_map, item_map, sasrec_dir)

    print("Preprocessing complete!")
    print(f"Total users: {len(user_map)}")
    print(f"Total items: {len(item_map)}")
    print(f"Training samples: {train_count}")
    print(f"Validation samples: {val_count}")
    print(f"Test samples: {test_count}")

# Run example
if __name__ == "__main__":
    improved_preprocessing(
        reviews_path='Electronics.json',
        meta_path='meta_Electronics.json',
        output_dir='dataset/sequential/Electronics',
        sasrec_dir='sasrec/map/Electronics',
        min_reviews=30,
        max_users=3000,
        min_records=30,
        max_records=50,
    )