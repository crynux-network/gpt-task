import time
from pathlib import Path
from huggingface_hub import snapshot_download

cache = Path(r"C:\Users\lency\Workspace\gpt-task\.models")
log = Path(r"C:\Users\lency\Workspace\gpt-task\examples\_dl_qwen35_27b.log")

def L(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

log.write_text("", encoding="utf-8")
attempt = 0
while True:
    attempt += 1
    try:
        L(f"snapshot attempt={attempt}")
        p = snapshot_download(
            "Qwen/Qwen3.5-27B",
            cache_dir=str(cache),
            max_workers=2,
            etag_timeout=60,
        )
        L(f"DONE {p}")
        break
    except Exception as e:
        L(f"FAIL {type(e).__name__}: {e}")
        time.sleep(5)
