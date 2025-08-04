#!/usr/bin/env python3
"""
Checkpoint download script for RunPod Custom Worker
Downloads checkpoint from Civitai if it doesn't exist locally
"""

import os
import requests
from pathlib import Path

def main():
    CHECKPOINT_URL = 'https://civitai.com/api/download/models/1590699?type=Model&format=SafeTensor&size=pruned&fp=fp16'
    CHECKPOINT_PATH = '/runpod-volume/models/checkpoints/jib_mix_illustrious_realistic_v2.safetensors'
    
    checkpoint_file = Path(CHECKPOINT_PATH)
    
    print('Checking checkpoint...')
    
    if not checkpoint_file.exists():
        print('Downloading checkpoint...')
        
        headers = {}
        civitai_key = os.environ.get('CIVITAI_API_KEY')
        if civitai_key:
            headers['Authorization'] = f'Bearer {civitai_key}'
        
        response = requests.get(CHECKPOINT_URL, headers=headers, stream=True)
        response.raise_for_status()
        
        with open(CHECKPOINT_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print('Checkpoint downloaded successfully!')
    else:
        print('Checkpoint already exists, skipping download.')

if __name__ == '__main__':
    main()
