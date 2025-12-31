
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
        Hardened Global-First Loader with File-System Safety.
        Ensures FileNotFoundError is handled if cache is cleared.
        """
        # 1. Sync ETags for the entire batch
        etags = {k: hashlib.md5(k.encode()).hexdigest() for k in keys}
        existing_etags = {k: etags[k] for k in keys if (self.cache_path / f"{etags[k]}.webp").exists()}

        # 2. GLOBAL AUTH & CACHE PROBE (No ML yet)
        url = f"{self.base_url.rstrip('/')}/get_multifiles?restore=false"
        # Probe with 1st key but send full ETag map for cache resolution
        payload = {"keys": [keys[0]], "etags": etags}
        
        resp = requests.post(url, json=payload, headers=self._get_headers())
        if resp.status_code != 200:
            raise Exception(f"Visdax Auth Failed: {resp.status_code}")

        data = resp.json()
        assets_map = {asset['key']: asset for asset in data.get("assets", [])}
        
        final_images_map = {} 
        keys_needing_restoration = []

        # 3. SORT: Valid Hits vs. Required Misses
        for key in keys:
            asset = assets_map.get(key)
            local_file = self.cache_path / f"{etags[key]}.webp"
            
            # CHECK: Does the file actually exist locally for a 304?
            is_locally_cached = local_file.exists()
            
            # CASE A: Ready to Materialize (Status 304/200 AND file is present or incoming)
            if asset and asset['status'] in [304, 200]:
                if asset['status'] == 200:
                    # Server is sending the 4K bytes
                    local_file.write_bytes(base64.b64decode(asset['content']))
                    is_locally_cached = True
                
                # FINAL SAFETY: Only attempt Image.open if file exists
                if is_locally_cached:
                    final_images_map[key] = np.array(Image.open(local_file).convert("RGB"))
                else:
                    # Server said 304, but file is gone! Force restoration.
                    keys_needing_restoration.append(key)
            else:
                # CASE B: Server confirms restoration is needed
                keys_needing_restoration.append(key)

        # 4. TARGETED PARALLEL RESTORATION
        if keys_needing_restoration:
            lumps = [keys_needing_restoration[i:i + lump_size] for i in range(0, len(keys_needing_restoration), lump_size)]
            lump_results = pqdm(lumps, lambda l: self._process_parallel_restoration(l, etags), n_jobs=n_jobs)
            
            for sub_map in lump_results:
                if sub_map:
                    final_images_map.update(sub_map)

        # 5. ORDERED RETURN
        return [final_images_map[k] for k in keys if k in final_images_map]
   
    def _process_parallel_restoration(self, lump, etags):
        """Worker thread for targeted ML restoration of 3 images."""
        url = f"{self.base_url.rstrip('/')}/get_multifiles?restore=true"
        try:
            # Targeted request for only the missing frames
            resp = requests.post(url, json={"keys": lump, "etags": {}}, headers=self._get_headers(), timeout=95)
            
            if resp.status_code == 200:
                return self._materialize_to_dict(resp.json(), lump, etags)
            
            # Recovery for Cloudflare 524
            if resp.status_code == 524:
                return {k: self.load(k) for k in lump}
        except:
            return {k: self.load(k) for k in lump}
        return {}
    def _materialize_to_dict(self, data, keys, etags):
        """Helper to ensure NumPy arrays are correctly mapped back to keys."""
        result_map = {}
        assets_map = {asset['key']: asset for asset in data.get("assets", [])}
        for key in keys:
            asset = assets_map.get(key)
            if asset and asset['status'] == 200:
                content = base64.b64decode(asset['content'])
                (self.cache_path / f"{etags[key]}.webp").write_bytes(content)
                img = Image.open(io.BytesIO(content)).convert("RGB")
                result_map[key] = np.array(img)
        return result_map



    

   
