
import os, requests, hashlib, base64, shutil, time
from pathlib import Path
from pqdm.threads import pqdm

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

    def _enforce_lru(self, incoming_size):
        """Strictly keeps the cache folder under 500MB."""
        files = sorted(self.cache_path.glob("*.webp"), key=os.path.getmtime)
        current_size = sum(f.stat().st_size for f in files)
        while current_size + incoming_size > self.limit and files:
            oldest = files.pop(0)
            current_size -= oldest.stat().st_size
            oldest.unlink()

    def load(self, key):
        """Single asset load with secret verification."""
        return self.load_batch([key])[0]

    def load_batch(self, keys):
        """
        The Core ML Function: Validates a batch of ETags.
        Ensures the secret is passed to the server for 'Compulsory Cache' checks.
        """
        etags = {k: hashlib.md5(k.encode()).hexdigest() for k in keys}
        existing_etags = {k: v for k, v in etags.items() if (self.cache_path / f"{v}.webp").exists()}

        payload = {"keys": keys, "etags": existing_etags}
        
        resp = requests.post(
            f"{self.base_url}/get_multifiles?restore=true", 
            json=payload, 
            headers=self._get_headers()
        )
        
        if resp.status_code != 200:
            if resp.status_code == 403:
                raise Exception("Visdax Access Denied: Missing or invalid Internal SDK Secret.")
            raise Exception(f"Visdax Batch Failed: {resp.status_code} - {resp.text}")

        data = resp.json()
        final_paths = []

        for asset in data.get("assets", []):
            key = asset['key']
            local_file = self.cache_path / f"{etags[key]}.webp"

            if asset['status'] == 304:
                # Cache Hit: Refresh time for LRU
                os.utime(local_file, None)
                final_paths.append(str(local_file))
            
            elif asset['status'] == 200:
                # Cache Miss: Download new high-fidelity data
                content = base64.b64decode(asset['content'])
                self._enforce_lru(len(content))
                local_file.write_bytes(content)
                final_paths.append(str(local_file))
            
            else:
                print(f"Visdax Error: Asset {key} failed with status {asset['status']}")

        return final_paths
