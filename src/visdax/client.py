
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
    def load_batch(self, keys, lump_size=2, n_jobs=2):
        etags = {k: hashlib.md5(k.encode()).hexdigest() for k in keys}
        
        # 1. PROBE: Metadata only check
        url = f"{self.base_url.rstrip('/')}/authorize_and_check_cache"
        resp = requests.post(url, json={"keys": keys, "etags": etags}, headers=self._get_headers())
        if resp.status_code != 200:
            raise Exception(f"Visdax Probe Failed: {resp.status_code}")

        assets_map = {asset['key']: asset for asset in resp.json().get("assets", [])}
        final_images_map = {} 
        keys_for_network = [] # Assets we need to actually download or restore

        # 2. INSTANT MATERIALIZATION (The "No pqdm" path)
        for key in keys:
            asset = assets_map.get(key)
            local_file = self.cache_path / f"{etags[key]}.webp"
            
            # If it's a 304 OR a 200 that we already have on disk, load it instantly
            if asset and asset['status'] in [304, 200] and local_file.exists():
                final_images_map[key] = np.array(Image.open(local_file).convert("RGB"))
            else:
                # This key is NOT on disk. It must go to the network.
                keys_for_network.append(key)

        # 3. CONDITIONAL PARALLEL RETRIEVAL
        # Only trigger pqdm if there is actually work to do over the network
        if keys_for_network:
            if len(keys_for_network) == 1:
                sub_map = self._process_parallel_lump(keys_for_network, etags)
			    if sub_map:
			        final_images_map.update(sub_map)
            else:
                lumps = [keys_for_network[i:i + lump_size] for i in range(0, len(keys_for_network), lump_size)]
                lump_results = pqdm(
                    lumps, 
                    lambda l: self._process_parallel_lump(l, etags), 
                    n_jobs=n_jobs,
                    desc="Parallel 4K Retrieval"
                )
                for sub_map in lump_results:
                    if sub_map: final_images_map.update(sub_map)

        return [final_images_map[k] for k in keys if k in final_images_map]
    def _process_parallel_lump(self, lump, etags):
        """Worker: Fetches data for 3 images. Server handles Restore vs Download internally."""
        url = f"{self.base_url.rstrip('/')}/get_multifiles?restore=true"
        try:
            # We don't send ETags here because we already know we need the data
            resp = requests.post(url, json={"keys": lump, "etags": {}}, headers=self._get_headers(), stream=True,timeout=95)
            
            if resp.status_code == 200:
                return self._materialize_multipart(resp.json(), lump, etags)
        except:
            raise RuntimeError(f"Visdax network failure for keys {lump}: {e}")
        return {}   
    def _materialize_multipart(self, resp, etags):
		"""
		Hardened multipart/mixed Visdax response parser.
		Returns: {key: np.ndarray}
		"""
		ctype = resp.headers.get("Content-Type", "")
		if "multipart/mixed" not in ctype:
			raise RuntimeError(f"Expected multipart response, got {ctype}")

		# ---- Extract boundary safely ----
		if "boundary=" not in ctype:
			raise RuntimeError("Multipart response missing boundary")

		boundary = ctype.split("boundary=", 1)[1].strip()
		boundary_bytes = b"--" + boundary.encode()

		buffer = b""
		current_meta = None
		result_map = {}

		def _process_part(part):
			nonlocal current_meta, result_map

			part = part.strip(b"\r\n")
			if not part:
				return

			# Must contain header/body separator
			if b"\r\n\r\n" not in part:
				logger.warning("Malformed multipart part (no header/body split)")
				return

			headers_raw, body = part.split(b"\r\n\r\n", 1)
			headers = headers_raw.decode(errors="ignore")

			# ---- JSON metadata ----
			if "application/json" in headers:
				try:
					meta = json.loads(body.decode())
				except Exception as e:
					logger.warning("Invalid JSON metadata: %s", e)
					current_meta = None
					return

				if "key" not in meta:
					logger.warning("Metadata missing 'key': %s", meta)
					current_meta = None
					return

				current_meta = meta
				return

			# ---- Image payload ----
			if "image/webp" in headers:
				if not current_meta:
					logger.warning("Image part received without preceding metadata")
					return

				key = current_meta.get("key")
				if key not in etags:
					logger.warning("Unknown key in image payload: %s", key)
					current_meta = None
					return

				etag = etags[key]
				path = self.cache_path / f"{etag}.webp"

				try:
					path.write_bytes(body)
					img = Image.open(io.BytesIO(body)).convert("RGB")
					result_map[key] = np.array(img)
				except Exception as e:
					logger.warning("Failed to materialize image for key=%s: %s", key, e)

				current_meta = None
				return

			# ---- Unknown content type ----
			logger.warning("Unknown multipart content type: %s", headers)

		# ---- Stream parsing loop ----
		for chunk in resp.iter_content(chunk_size=16384):
			if not chunk:
				continue

			buffer += chunk

			while True:
				idx = buffer.find(boundary_bytes)
				if idx == -1:
					# Keep tail in case boundary is split across chunks
					if len(buffer) > len(boundary_bytes):
						buffer = buffer[-len(boundary_bytes):]
					break

				part = buffer[:idx]
				buffer = buffer[idx + len(boundary_bytes):]
				_process_part(part)

		# ---- Flush any remaining buffered data ----
		if buffer.strip():
			_process_part(buffer)

		# ---- Final sanity check ----
		if current_meta is not None:
			logger.warning("Dangling metadata without image payload: %s", current_meta)

		return result_map

    




    

   
