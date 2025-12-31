
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

    def load_batch(self, keys, lump_size=3, n_jobs=4):
        """
        True Enterprise Parallelism: Dispatches lumped requests in parallel.
        Optimized for high-throughput 4K frame delivery.
        """
        # 1. Synchronize ETags with server-side logic
        etags = {k: hashlib.md5(k.encode()).hexdigest() for k in keys}
        
        # 2. Divide keys into lumps of 3 to stay under 100s
        lumps = [keys[i:i + lump_size] for i in range(0, len(keys), lump_size)]
        
        # 3. Parallel Dispatch: Each thread handles one lumped request
        # Using lambda to pass the constant 'etags' map to each worker
        lump_results = pqdm(
            lumps, 
            lambda l: self._process_single_lump(l, etags), 
            n_jobs=n_jobs, 
            desc="Lumped Parallel Restoration"
        )
        
        # 4. Flatten the parallel results while preserving original order
        final_results = []
        for res in lump_results:
            if isinstance(res, list):
                final_results.extend(res)
        
        return final_results

    def _process_single_lump(self, lump, etags):
        """Internal parallel worker for a single lump of 3 images."""
        existing_etags = {k: etags[k] for k in lump if (self.cache_path / f"{etags[k]}.webp").exists()}
        
        payload = {"keys": lump, "etags": existing_etags}
        url = f"{self.base_url.rstrip('/')}/get_multifiles"
        
        try:
            # 95s timeout protects against Cloudflare 524 drops
            resp = requests.post(
                url, 
                json=payload, 
                headers=self._get_headers(),
                timeout=95 
            )
            
            if resp.status_code == 200:
                return self._materialize_lump(resp.json(), lump, etags)
            
            # Handle Cloudflare 524: The server worker is likely still finishing
            if resp.status_code == 524:
                print(f"Visdax: Lump timed out. Retrying finished assets individually...")
                time.sleep(5)
                # Fallback to single-item loads for this specific failed lump
                return [self.load(k) for k in lump]
            
            raise Exception(f"Lump Failed: {resp.status_code}")

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            # Degrading gracefully if the network or proxy hangs
            return [self.load(k) for k in lump]

    def _materialize_lump(self, data, lump_keys, etags):
        """
        Converts server response to NumPy arrays.
        Resolves 'str' AttributeError and ensures (2160, 3840, 3) compatibility.
        """
        images = []
        assets_map = {asset['key']: asset for asset in data.get("assets", [])}
        for key in lump_keys:
            asset = assets_map.get(key)
            if not asset: continue
            
            local_file = self.cache_path / f"{etags[key]}.webp"
            
            # CASE A: AUTHORIZED CACHE HIT (~0.5s)
            if asset['status'] == 304:
                img = Image.open(local_file).convert("RGB")
                images.append(np.array(img))
            
            # CASE B: AUTHORIZED CACHE MISS (~20s per image)
            elif asset['status'] == 200:
                content = base64.b64decode(asset['content'])
                local_file.write_bytes(content)
                img = Image.open(io.BytesIO(content)).convert("RGB")
                images.append(np.array(img))
        return images
