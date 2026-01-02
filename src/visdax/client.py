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
                raise ValueError("Load called with an empty list.")
            key = key[0]
            print(f"Visdax Warning: .load() received a list. Processing first item: {key}")

        results = self.load_batch([key])

        if not results:
            raise Exception(f"Visdax Error: Failed to load asset '{key}'.")

        return results[0]

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
        Hardened multipart/mixed Visdax response parser.
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
        current_meta = None
        result_map = {}

        def _process_part(part):
            nonlocal current_meta, result_map

            part = part.strip(b"\r\n")
            if not part:
                return

            if b"\r\n\r\n" not in part:
                logger.warning("Malformed multipart part")
                return

            headers_raw, body = part.split(b"\r\n\r\n", 1)
            headers = headers_raw.decode(errors="ignore")

            if "application/json" in headers:
                try:
                    meta = json.loads(body.decode())
                except Exception as e:
                    logger.warning("Invalid JSON metadata: %s", e)
                    current_meta = None
                    return

                if "key" not in meta:
                    logger.warning("Metadata missing key")
                    current_meta = None
                    return

                current_meta = meta
                return

            if "image/webp" in headers:
                if not current_meta:
                    logger.warning("Image without metadata")
                    return

                key = current_meta.get("key")
                if key not in etags:
                    logger.warning("Unknown key %s", key)
                    current_meta = None
                    return

                path = self.cache_path / f"{etags[key]}.webp"

                try:
                    path.write_bytes(body)
                    img = Image.open(io.BytesIO(body)).convert("RGB")
                    result_map[key] = np.array(img)
                except Exception as e:
                    logger.warning("Image decode failed: %s", e)

                current_meta = None
                return

            logger.warning("Unknown multipart content")

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

        if buffer.strip():
            _process_part(buffer)

        if current_meta is not None:
            logger.warning("Dangling metadata: %s", current_meta)

        return result_map
