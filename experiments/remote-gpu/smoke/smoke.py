import subprocess, urllib.request
print(subprocess.run(["nvidia-smi","-L"],capture_output=True,text=True).stdout or "NO GPU")
try:
    urllib.request.urlopen("https://huggingface.co",timeout=10); print("INTERNET OK")
except Exception as e: print("INTERNET FAIL",e)
import torch; print("torch",torch.__version__,torch.cuda.is_available())
