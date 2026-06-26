"""SSH-VM backend for the autotune probe (Paperspace, Lambda Labs, any SSH GPU box).

Uploads `probe_runner.py` + the config to the VM, runs it over SSH, and returns the same result
JSON the Modal backend returns. Connection and secrets come entirely from env:

  AUTOTUNE_SSH_HOST              (required)  e.g. 184.105.190.154
  AUTOTUNE_SSH_USER              (required)  e.g. paperspace / ubuntu
  AUTOTUNE_SSH_PASSWORD          (one of)    password auth (Paperspace)
  AUTOTUNE_SSH_KEY               (one of)    path to a private key (Lambda Labs)
  AUTOTUNE_SSH_ACTIVATE          (optional)  enter the swift env, e.g.
                                             'source ~/envs/swift-env/bin/activate'
  AUTOTUNE_REMOTE_DIR            (optional)  default /tmp/autotune
  AUTOTUNE_CUDA_VISIBLE_DEVICES  (optional)  default 0..N-1 from hardware_number
  HF_TOKEN                       (optional)  forwarded so swift can pull the model/dataset
"""
import json
import os
import posixpath


def _echo(config):
    bs = config.get("per_device_train_batch_size", 1)
    eff = config.get("effective_batch")
    n = int(config.get("hardware_number", 1))
    denom = bs * n
    ga = (eff // denom) if (denom and eff and eff % denom == 0) else None   # mirrors probe_runner
    return {"model": config.get("model"), "bs": bs, "ga": ga, "world_size": n,
            "effective_batch": eff, "attn_impl": config.get("attn_impl", "flash_attn")}


def _connect():
    import paramiko
    host = os.environ["AUTOTUNE_SSH_HOST"]
    user = os.environ["AUTOTUNE_SSH_USER"]
    key = os.environ.get("AUTOTUNE_SSH_KEY")
    password = os.environ.get("AUTOTUNE_SSH_PASSWORD")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if key:
        client.connect(host, username=user, key_filename=os.path.expanduser(key), timeout=30)
    elif password:
        client.connect(host, username=user, password=password, timeout=30)
    else:
        raise RuntimeError("set AUTOTUNE_SSH_KEY (key auth) or AUTOTUNE_SSH_PASSWORD (password auth)")
    return client


def _run_remote(config: dict, remote_tail: str) -> dict:
    """Shared SSH plumbing: upload probe_runner.py + config.json, run `remote_tail` in the remote dir
    (behind the activate prefix), stream output, return the parsed result.json (or an error dict)."""
    try:
        import paramiko  # noqa: F401
    except ImportError:
        return {"status": "error", "config": _echo(config),
                "error_detail": "SSH backend needs paramiko — `pip install paramiko`"}

    for var in ("AUTOTUNE_SSH_HOST", "AUTOTUNE_SSH_USER"):
        if not os.environ.get(var):
            return {"status": "error", "config": _echo(config),
                    "error_detail": f"missing required env var {var}"}

    remote_dir = os.environ.get("AUTOTUNE_REMOTE_DIR", "/tmp/autotune")
    activate = os.environ.get("AUTOTUNE_SSH_ACTIVATE", "").strip()
    here = os.path.dirname(os.path.abspath(__file__))

    try:
        client = _connect()
    except Exception as e:
        return {"status": "error", "config": _echo(config), "error_detail": f"ssh connect failed: {e}"}

    try:
        sftp = client.open_sftp()
        try:
            sftp.mkdir(remote_dir)
        except IOError:
            pass                                              # already exists
        sftp.put(os.path.join(here, "probe_runner.py"), posixpath.join(remote_dir, "probe_runner.py"))
        with sftp.open(posixpath.join(remote_dir, "config.json"), "w") as f:
            f.write(json.dumps(config))

        prefix = f"{activate} && " if activate else ""
        cmd = f"cd {remote_dir} && {prefix}{remote_tail}"

        _stdin, stdout, _stderr = client.exec_command(cmd, get_pty=True)
        for line in iter(stdout.readline, ""):                # stream remote output live
            print(line, end="")
        rc = stdout.channel.recv_exit_status()

        try:
            with sftp.open(posixpath.join(remote_dir, "result.json")) as f:
                return json.loads(f.read().decode())
        except Exception as e:
            return {"status": "error", "config": _echo(config),
                    "error_detail": f"remote produced no result.json (exit {rc}): {e}"}
    finally:
        client.close()


def run_probe(config: dict) -> dict:
    """Run one probe on the SSH VM. Returns the result dict, or a status='error' dict on failure."""
    n_gpu = int(config.get("hardware_number", 1))      # default visible devices to 0..N-1 (multi-GPU)
    cvd = os.environ.get("AUTOTUNE_CUDA_VISIBLE_DEVICES") or ",".join(str(i) for i in range(n_gpu))
    hf = os.environ.get("HF_TOKEN", "")
    envs = f"CUDA_VISIBLE_DEVICES={cvd} USE_HF=1" + (f" HF_TOKEN={hf}" if hf else "")
    return _run_remote(config, f"{envs} python probe_runner.py config.json result.json")


def scan_env(config: dict) -> dict:
    """Run the no-training capability scan on the SSH VM (same box the probe will use)."""
    return _run_remote(config, "python probe_runner.py --scan result.json")
