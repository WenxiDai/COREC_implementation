import json
user_id_map = 'sasrec/map/Beauty/user_map.json'
item_id_map = 'sasrec/map/Beauty/item_map.json'

with open(user_id_map, 'r', encoding='utf-8') as f:
    user_map_dict = json.load(f)          # {"A3SJ7ZAQBHY76": 1, ...}
with open(item_id_map, 'r', encoding='utf-8') as f:
    item_map_dict = json.load(f)          # {"B00005NTRC": 14362, ...}

# idx ➜ id
user_idx2id = {int(idx): uid for uid, idx in user_map_dict.items()}
item_idx2id = {int(idx): iid for iid, idx in item_map_dict.items()}

in_path  = 'sas_results/beauty/candidates.txt'          
out_path = 'candidates_in_sas/beauty/candidates_mapped.txt'     

with open(in_path, 'r', encoding='utf-8') as fin, \
     open(out_path, 'w', encoding='utf-8') as fout:


    for line in fin:
        parts = line.strip().split()
        if not parts:
            continue
        u_idx      = int(parts[0])
        window_idx = parts[1]                 
        item_idxs  = map(int, parts[2:])      

        u_id     = user_idx2id.get(u_idx, f'UNK_{u_idx}')
        item_ids = [item_idx2id.get(i, f'UNK_{i}') for i in item_idxs]

        fout.write(' '.join([u_id, window_idx] + item_ids) + '\n')