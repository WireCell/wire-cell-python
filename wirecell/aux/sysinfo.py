import subprocess

def cpu():
    nproc = 0;
    model = ""
    for line in open("/proc/cpuinfo").readlines():
        if line.startswith("processor"):
            nproc += 1
        if line.startswith("model name"):
            model = line.split(":")[1]
    return dict(nproc=nproc, model=model.strip())
def gpu():
    gpus = list()
    for line in subprocess.check_output(["nvidia-smi","-L"]).decode().split("\n"):
        line = line.strip()
        if not line:
            continue
        name = line.split(":")[1].split("(")[0]
        gpus.append(name.strip())
    return gpus

def asdict():
    return dict(cpu=cpu(), gpus=gpu())
