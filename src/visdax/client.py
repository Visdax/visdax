
import os, requests, hashlib, base64, shutil, time,io
from pathlib import Path
from pqdm.threads import pqdm
from PIL import Image

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

    def load_batch(self, keys):
        """
        Hardened Batch ML Function.
        Derives identity from session token to return NumPy arrays for .shape compatibility.
        """
        # 1. GENERATE ETags 
        # We use the raw key for the ETag to match a simplified server-side check.
        etags = {k: hashlib.md5(k.encode()).hexdigest() for k in keys}
        
        # Check local cache for existing .webp files
        existing_etags = {k: v for k, v in etags.items() if (self.cache_path / f"{v}.webp").exists()}

        payload = {"keys": keys, "etags": existing_etags}
        
        resp = requests.post(
            f"{self.base_url}/get_multifiles?restore=true", 
            json=payload, 
            headers=self._get_headers(), # Identity is carried in these headers
            timeout=1200
        )
        
        if resp.status_code != 200:
            if resp.status_code == 403:
                raise Exception("Visdax Access Denied: Missing or invalid Internal SDK Secret.")
            raise Exception(f"Visdax Batch Failed: {resp.status_code} - {resp.text}")

        data = resp.json()
        final_images = [] # Returning NumPy objects to fix 'str' AttributeError

        # Map results for order preservation
        assets_map = {asset['key']: asset for asset in data.get("assets", [])}

        for key in keys:
            asset = assets_map.get(key)
            if not asset:
                continue
                
            local_file = self.cache_path / f"{etags[key]}.webp"

            # CASE A: CACHE HIT (304) - Load from disk to NumPy
            if asset['status'] == 304:
                os.utime(local_file, None) # Refresh LRU timestamp for cache management
                img = Image.open(local_file).convert("RGB")
                final_images.append(np.array(img))
            
            # CASE B: CACHE MISS (200) - Download and convert
            elif asset['status'] == 200:
                content = base64.b64decode(asset['content'])
                #self._enforce_lru(len(content)) # Ensure cache doesn't exceed limits
                local_file.write_bytes(content)
                
                # Convert raw bytes directly to NumPy array
                img = Image.open(io.BytesIO(content)).convert("RGB")
                final_images.append(np.array(img))
            
            else:
                print(f"Visdax Error: Asset {key} failed with status {asset['status']}")

        return final_images 
