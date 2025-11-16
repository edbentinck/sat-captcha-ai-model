import os
import hashlib
import time
import requests

IMAGE_URL = "https://www.inm.gob.mx/spublic/captchas"
OUTPUT_DIR = "dataset"
TOTAL_IMAGES = 1_000
DELAY_SECONDS = 0.2  # be nice to the server

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_images():
    for i in range(TOTAL_IMAGES):
        try:
            resp = requests.get(IMAGE_URL, timeout=10)
            resp.raise_for_status()

            # Basic sanity check: is this an image?
            content_type = resp.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                print(f"[{i}] Skipped – not an image. Content-Type: {content_type}")
                continue

            # Save as a numbered file
            filename = os.path.join(OUTPUT_DIR, f"image_{i:05d}.png")
            with open(filename, "wb") as f:
                f.write(resp.content)

            print(f"[{i}] Saved {filename}")

            # Avoid hammering the server
            time.sleep(DELAY_SECONDS)

        except Exception as e:
            print(f"[{i}] Error: {e}")

def file_hash(path, chunk_size=8192):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def remove_duplicates():
    seen_hashes = {}
    deleted_files = []

    for root, _, files in os.walk(OUTPUT_DIR):
        for name in files:
            path = os.path.join(root, name)

            try:
                h = file_hash(path)
            except Exception as e:
                print(f"Error hashing {path}: {e}")
                continue

            if h in seen_hashes:
                # Duplicate found – delete it
                print(f"Duplicate detected: {path} (same as {seen_hashes[h]})")
                try:
                    os.remove(path)
                    deleted_files.append(path)
                except Exception as e:
                    print(f"Error deleting {path}: {e}")
            else:
                seen_hashes[h] = path

    print(f"\nDone. Deleted {len(deleted_files)} duplicate files.")

if __name__ == "__main__":
    download_images()
    remove_duplicates()
