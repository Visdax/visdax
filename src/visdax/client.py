
import os, requests, hashlib, base64, shutil, time,io
from pathlib import Path
from pqdm.threads import pqdm
from PIL import Image
import numpy as np

class VisdaxClient:
    def __init__(self, api_key, project, limit_mb=500):
        self.api_key = api_key
        self.project = project
        self.limit = limit_mb * 1024 * 1024
        self.cache_path = Path("~/.visdax_cache").expanduser()
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://www.visdax.com/api/v1"
        
        # Reads the secret directly from the environment where the SDK is running
        

    def _get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Visdax-Project": self.project,
        }

    # ==========================================
    # 1. SUBMISSION (UPLOAD) FUNCTIONS
    # ==========================================

    def submit(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f)}
            resp = requests.post(
                f"{self.base_url}/post_file", 
                headers=self._get_headers(), 
                files=files
            )
    
        # Debugging: If not a 200 OK, print the raw response
        if resp.status_code != 200:
            print(f"Server Error {resp.status_code}: {resp.text}")
            resp.raise_for_status() # This will give a better traceback
        
        return resp.json()

    def submit_batch(self, file_paths, n_jobs=4):
        """Parallel upload for large datasets."""
        return pqdm(file_paths, self.submit, n_jobs=n_jobs)

    # ==========================================
    # 2. RETRIEVAL (DOWNLOAD + LRU) FUNCTIONS
    # ==========================================

 #   def _enforce_lru(self, incoming_size):
 #       """Strictly keeps the cache folder under 500MB."""
 #       files = sorted(self.cache_path.glob("*.webp"), key=os.path.getmtime)
 #       current_size = sum(f.stat().st_size for f in files)
 #       while current_size + incoming_size > self.limit and files:
 #           oldest = files.pop(0)
 #           current_size -= oldest.stat().st_size
 #           oldest.unlink()

    def load(self, key):
        """
        Patched Single Asset Load.
        Ensures 'key' is a string to prevent AttributeError: 'list' object has no attribute 'encode'.
        """
        if isinstance(key, list):
            # Gracefully handle accidental list input by taking the first item
            if not key:
                raise ValueError("Load called with an empty list.")
            key = key[0]
            print(f"Visdax Warning: .load() received a list. Processing first item: {key}")

        results = self.load_batch([key])
    
        # Check if results list is empty to prevent 'IndexError: list index out of range'
        if not results:
            raise Exception(f"Visdax Error: Failed to load asset '{key}'. Check server logs.")
        
        return results[0]

    def load_batch(self, keys, lump_size=3):
        """
        Lumped Chunking Strategy for Enterprise Delivery.
        Groups 3 assets per request to stay under 100s Cloudflare limit.
        """
        # 1. Synchronize ETags with server-side logic
        etags = {k: hashlib.md5(k.encode()).hexdigest() for k in keys}
        
        # 2. Divide into lumps of 3
        lumps = [keys[i:i + lump_size] for i in range(0, len(keys), lump_size)]
        
        final_results = []

        for lump in lumps:
            # Check local cache for these specific 3 keys
            existing_etags = {k: etags[k] for k in lump if (self.cache_path / f"{etags[k]}.webp").exists()}
            
            payload = {"keys": lump, "etags": existing_etags}
            url = f"{self.base_url.rstrip('/')}/get_multifiles"
            
            try:
                # 95s timeout to catch issues before Cloudflare drops the connection
                resp = requests.post(
                    url, 
                    json=payload, 
                    headers=self._get_headers(),
                    timeout=95 
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    # Materialize lump into NumPy arrays
                    lump_images = self._materialize_lump(data, lump, etags)
                    final_results.extend(lump_images)
                
                elif resp.status_code == 524:
                    # Cloudflare timeout means the server worker is still running
                    print(f"Visdax: Lump timed out (Cloudflare 524). Retrying individually...")
                    time.sleep(5)
                    # Recovery: Shred this lump into individual requests
                    final_results.extend(self.load_batch(lump, lump_size=1))
                else:
                    raise Exception(f"Visdax Error {resp.status_code}: {resp.text}")

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                # Fallback to single-item requests if the 3-image lump is too heavy
                final_results.extend(self.load_batch(lump, lump_size=1))

        return final_results

    def _materialize_lump(self, data, lump_keys, etags):
        """Helper to convert server JSON lump into NumPy array list."""
        images = []
        assets_map = {asset['key']: asset for asset in data.get("assets", [])}
        for key in lump_keys:
            asset = assets_map.get(key)
            if not asset: continue
            
            local_file = self.cache_path / f"{etags[key]}.webp"
            # Authorized Cache Hit (~0.5s Path)
            if asset['status'] == 304:
                img = Image.open(local_file).convert("RGB")
                images.append(np.array(img))
            # Authorized Cache Miss (~20s per image Path)
            elif asset['status'] == 200:
                content = base64.b64decode(asset['content'])
                local_file.write_bytes(content)
                img = Image.open(io.BytesIO(content)).convert("RGB")
                images.append(np.array(img))
        return images
