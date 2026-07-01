import os
import subprocess

env = os.environ.copy()
if os.path.isfile("/content/colab_env.sh"):
    with open("/content/colab_env.sh", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.removeprefix("export ").strip()
            val = val.strip().strip("'\"")
            if key:
                env[key] = val

r = subprocess.run(
    ["bash", "-x", "/content/colab_locomo_hf.sh"],
    capture_output=True,
    text=True,
    env=env,
    timeout=600,
)
print("=== stdout tail ===")
print(r.stdout[-12000:])
print("=== stderr tail ===")
print(r.stderr[-12000:])
print("exit", r.returncode)
