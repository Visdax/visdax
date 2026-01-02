import os, requests, hashlib, base64, shutil, time, io
from pathlib import Path
from pqdm.threads import pqdm
from PIL import Image
import numpy as np
import json
import logging

logger = logging.getLogger("visdax")


class VisdaxClient:
    def __init__(self, api_key, project, limit_mb=500):
        self.api_key = api_key
        self.project = project
        self.limit = limit_mb * 1024 * 1024
        self.cache_path = Path("~/.visdax_cache").expanduser()
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://www.visdax.com/api/v1"

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

        if resp.status_code != 200:
            print(f"Server Error {resp.status_code}: {resp.text}")
            resp.raise_for_status()

        return resp.json()

    def submit_batch(self, file_paths, n_jobs=4):
        return pqdm(file_paths, self.submit, n_jobs=n_jobs)

    # ==========================================
    # 2. RETRIEVAL (DOWNLOAD + LRU) FUNCTIONS
    # ==========================================

    def load(self, key):
		if isinstance(key, list):
			if not key:
				raise ValueError("load() called with empty list")
			logger.warning("load() received list, using first element")
			key = key[0]

		etag = hashlib.md5(key.encode()).hexdigest()
		local_file = self.cache_path / f"{etag}.webp"

		# ---- Local cache hit ----
		if local_file.exists():
			try:
				return np.array(Image.open(local_file).convert("RGB"))
			except Exception:
				pass  # fall through to network

		# ---- Network fetch (single-file endpoint) ----
		url = f"{self.base_url.rstrip('/')}/get_file?restore=true"

		resp = requests.post(
			url,
			json={
				"keys": [key],
				"etags": {key: etag}
			},
			headers=self._get_headers(),
			stream=True,
			timeout=95
		)

		if resp.status_code != 200:
			raise Exception(f"Visdax load failed for {key}: {resp.status_code}")

		# ---- SDK-level 304 ----
		if resp.headers.get("Content-Type", "").startswith("application/json"):
			payload = resp.json()
			if payload.get("status") == 304:
				if local_file.exists():
					return np.array(Image.open(local_file).convert("RGB"))
				raise Exception(f"Visdax cache inconsistency for {key}")

		# ---- Raw image bytes ----
		data = resp.content

		try:
			local_file.write_bytes(data)
			img = Image.open(io.BytesIO(data)).convert("RGB")
			return np.array(img)
		except Exception as e:
			raise Exception(f"Image decode failed for {key}: {e}")

    def load_batch(self, keys, lump_size=2, n_jobs=2):
        etags = {k: hashlib.md5(k.encode()).hexdigest() for k in keys}

        url = f"{self.base_url.rstrip('/')}/authorize_and_check_cache"
        resp = requests.post(
            url,
            json={"keys": keys, "etags": etags},
            headers=self._get_headers()
        )

        if resp.status_code != 200:
            raise Exception(f"Visdax Probe Failed: {resp.status_code}")

        assets_map = {a["key"]: a for a in resp.json().get("assets", [])}
        final_images_map = {}
        keys_for_network = []

        for key in keys:
            asset = assets_map.get(key)
            local_file = self.cache_path / f"{etags[key]}.webp"

            if asset and asset["status"] in (304, 200) and local_file.exists():
                final_images_map[key] = np.array(
                    Image.open(local_file).convert("RGB")
                )
            else:
                keys_for_network.append(key)

        if keys_for_network:
            if len(keys_for_network) == 1:
                sub_map = self._process_parallel_lump(keys_for_network, etags)
                if sub_map:
                    final_images_map.update(sub_map)
            else:
                lumps = [
                    keys_for_network[i:i + lump_size]
                    for i in range(0, len(keys_for_network), lump_size)
                ]
                lump_results = pqdm(
                    lumps,
                    lambda l: self._process_parallel_lump(l, etags),
                    n_jobs=n_jobs,
                    desc="Parallel 4K Retrieval"
                )
                for sub_map in lump_results:
                    if sub_map:
                        final_images_map.update(sub_map)

        return [final_images_map[k] for k in keys if k in final_images_map]

    def _process_parallel_lump(self, lump, etags):
        url = f"{self.base_url.rstrip('/')}/get_multifiles?restore=true"
        try:
            resp = requests.post(
                url,
                json={"keys": lump, "etags": {}},
                headers=self._get_headers(),
                stream=True,
                timeout=95
            )

            if resp.status_code == 200:
                return self._materialize_multipart(resp, etags)
        except Exception as e:
            raise RuntimeError(
                f"Visdax network failure for keys {lump}: {e}"
            )
        return {}

    def _materialize_multipart(self, resp, etags):
		"""
		RFC-correct multipart/mixed Visdax parser.
		One part == one asset.
		Metadata is carried in headers.
		Returns: {key: np.ndarray}
		"""

		ctype = resp.headers.get("Content-Type", "")
		if "multipart/mixed" not in ctype:
			raise RuntimeError(f"Expected multipart response, got {ctype}")

		if "boundary=" not in ctype:
			raise RuntimeError("Multipart response missing boundary")

		boundary = ctype.split("boundary=", 1)[1].strip()
		boundary_bytes = b"--" + boundary.encode()

		buffer = b""
		result_map = {}

		def _process_part(part: bytes):
			if not part or part in (b"--",):
				return

			# DO NOT strip bytes — multipart framing is byte-exact
			if b"\r\n\r\n" not in part:
				logger.warning("Malformed multipart part (no header/body separator)")
				return

			headers_raw, body = part.split(b"\r\n\r\n", 1)
			headers_text = headers_raw.decode(errors="ignore")

			headers = {}
			for line in headers_text.split("\r\n"):
				if ":" in line:
					k, v = line.split(":", 1)
					headers[k.strip().lower()] = v.strip()

			content_type = headers.get("content-type", "")
			if not content_type.startswith("image/"):
				logger.warning("Skipping non-image multipart part: %s", content_type)
				return

			key = headers.get("x-visdax-key")
			if not key:
				logger.warning("Image part missing X-Visdax-Key")
				return

			if key not in etags:
				logger.warning("Unknown key %s", key)
				return

			try:
				img = Image.open(io.BytesIO(body)).convert("RGB")
				result_map[key] = np.array(img)
			except Exception as e:
				logger.warning("Image decode failed for %s: %s", key, e)

		for chunk in resp.iter_content(chunk_size=16384):
			if not chunk:
				continue

			buffer += chunk

			while True:
				idx = buffer.find(boundary_bytes)
				if idx == -1:
					buffer = buffer[-len(boundary_bytes):]
					break

				part = buffer[:idx]
				buffer = buffer[idx + len(boundary_bytes):]
				_process_part(part)

		if buffer.strip(b"\r\n-"):
			_process_part(buffer)

		return result_map
