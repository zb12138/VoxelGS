import re
import pandas as pd
import glob
import json
import os 
import sys
def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        log_content = file.read()
    
    matches = reversed(log_content.split('\n')[-5:-1])
    
    data = []
    for match in matches:
        try:
            data.append(flatten_dict(json.loads(match.split('|')[-1].replace("'",'"'))))
        except json.JSONDecodeError:
            print(f"Error decoding JSON from log: {match}")
    # merge into one dict
    if len(data) > 1:
        merged_data = {}
        for d in data:
            merged_data.update(d)
    elif len(data) == 1:
        merged_data = data[0]
    return merged_data

def stat_log(log_dir = 'output/voxelScaffold/Nerf_Synthetic'):
    log_files = glob.glob(f'{log_dir}/*/test.log')  
    all_data = []
    
    for log_file in log_files:
        log_data = {'name': os.path.basename(os.path.dirname(log_file))} 
        log_data.update(parse_log_file(log_file))
        all_data.append(log_data)

    df = pd.DataFrame(all_data)
    # add means
    mean_row = df.mean(numeric_only=True)
    mean_row['name'] = 'mean'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    #
    # save to txt
    df.to_csv(f'{log_dir}/{os.path.basename(log_dir)}_results_summary.csv', index=False, sep=',')

if __name__ == "__main__":
    stat_log(log_dir = sys.argv[1])