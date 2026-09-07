"""
Standalone weight-download script.
Called by entrypoint.sh before gunicorn starts.

Uses huggingface_hub for robust downloads (resumable, handles auth, retries).
Downloads only files that are not already present on disk — fully idempotent.
"""
import os
import sys

from huggingface_hub import hf_hub_download

# ── Configuration ──────────────────────────────────────────────────────────────
# Set HF_REPO_ID in docker-compose / environment.
# Format: "username/repo-name"  e.g. "piyush-s-deshmukh/change_detection_loc_and_cls"
REPO_ID = os.getenv("HF_REPO_ID", "").strip()
if not REPO_ID:
    print("ERROR: HF_REPO_ID environment variable is not set.")
    print("       Set it in docker-compose.yml  e.g.  HF_REPO_ID=username/repo-name")
    sys.exit(1)

# Optional: set HF_TOKEN for private repos
HF_TOKEN = os.getenv("HF_TOKEN", None) or None   # None → public access

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_FOLDER = os.path.join(BASE_DIR, "DPN92_weights")
# NUM_FOLDS      = 3   # <- was: download all 3 folds (~2.13 GB total)
NUM_FOLDS      = 1      # Only download the seed-0 LOC + CLS weights (~725 MB),
                        # matching NUM_FOLDS in app_dpn92.py

# Files to download:  filename in HF repo  →  local destination path
WEIGHT_FILES = {
    **{
        f"LOC/dpn92_loc_{i}_0_best.pth":
            os.path.join(WEIGHTS_FOLDER, "LOC", f"dpn92_loc_{i}_0_best.pth")
        for i in range(NUM_FOLDS)
    },
    **{
        f"CLS/BEST/dpn92_cls_cce_{i}_1_best":
            os.path.join(WEIGHTS_FOLDER, "CLS", "BEST", f"dpn92_cls_cce_{i}_1_best")
        for i in range(NUM_FOLDS)
    },
}
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("DPN92 Weight Check")
print(f"  Repo  : {REPO_ID}")
print(f"  Local : {WEIGHTS_FOLDER}")
print(f"  Auth  : {'token provided' if HF_TOKEN else 'no token (public repo)'}")
print("=" * 60)

any_downloaded = False

for hf_filename, local_path in WEIGHT_FILES.items():
    # ── Already on disk — skip ────────────────────────────────────────────────
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"  ✓ Present ({size_mb:.0f} MB): {hf_filename}")
        continue

    # ── Download via huggingface_hub ──────────────────────────────────────────
    print(f"  ↓ Downloading: {hf_filename} ...")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=hf_filename,
            local_dir=WEIGHTS_FOLDER,
            token=HF_TOKEN,
        )
        # hf_hub_download may put the file in a cache subdir; move it if needed
        if os.path.abspath(downloaded) != os.path.abspath(local_path):
            import shutil
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            shutil.move(downloaded, local_path)

        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"  ✓ Done ({size_mb:.0f} MB): {hf_filename}")
        any_downloaded = True

    except Exception as exc:
        # Remove any partial file so a retry starts clean
        if os.path.exists(local_path):
            os.remove(local_path)
        print(f"\n  ✗ FAILED to download: {hf_filename}")
        print(f"    Error: {exc}")
        print()
        print("  Troubleshooting:")
        print(f"    1. Make sure files are uploaded to: https://huggingface.co/{REPO_ID}")
        print( "    2. Expected files in the repo:")
        for f in WEIGHT_FILES:
            print(f"         {f}")
        print( "    3. For private repos, set HF_TOKEN in docker-compose environment.")
        sys.exit(1)

print("=" * 60)
if any_downloaded:
    print("All missing weights downloaded successfully.")
else:
    print("All weights already present — no downloads needed.")
print("=" * 60)