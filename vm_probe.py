"""SSH-VM backend for the autotune probe (Paperspace, Lambda Labs, any SSH GPU box).

Uploads `probe_runner.py` + the config to the VM, runs it over SSH, and returns the SAME Layer-1
JSON the Modal backend returns. Connection + secrets come entirely from env, so the user just
exports them and picks the backend — no code changes:

  AUTOTUNE_SSH_HOST              (required)  e.g. 184.105.190.154
  AUTOTUNE_SSH_USER              (required)  e.g. paperspace / ubuntu
  AUTOTUNE_SSH_PASSWORD          (one of)    password auth (Paperspace)
  AUTOTUNE_SSH_KEY               (one of)    path to a private key (Lambda Labs)
  AUTOTUNE_SSH_ACTIVATE          (optional)  enter the swift env, e.g.
                                             'source ~/envs/swift-env/bin/activate'
  AUTOTUNE_REMOTE_DIR            (optional)  default /tmp/autotune
  AUTOTUNE_CUDA_VISIBLE_DEVICES  (optional)  default '0'
  HF_TOKEN                       (optional)  forwarded so swift can pull the model/dataset
"""
import json
import os
import posixpath


def _echo(config):
    bs = config.get("per_device_train_batch_size")
    ga = config.get("gradient_accumulation_steps")
    return {"model": config.get("model"), "bs": bs, "ga": ga,
            "effective_batch": (bs * ga) if bs and ga else None,
            "attn_impl": config.get("attn_impl")}


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


def run_probe(config: dict) -> dict:
    """Run one probe on the SSH VM. Returns the Layer-1 dict, or a status='error' dict on failure."""
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
    cvd = os.environ.get("AUTOTUNE_CUDA_VISIBLE_DEVICES", "0")
    hf = os.environ.get("HF_TOKEN", "")
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

        envs = f"CUDA_VISIBLE_DEVICES={cvd} USE_HF=1" + (f" HF_TOKEN={hf}" if hf else "")
        prefix = f"{activate} && " if activate else ""
        cmd = f"cd {remote_dir} && {prefix}{envs} python probe_runner.py config.json result.json"

        _stdin, stdout, _stderr = client.exec_command(cmd, get_pty=True)
        for line in iter(stdout.readline, ""):                # stream remote swift output live
            print(line, end="")
        rc = stdout.channel.recv_exit_status()

        try:
            with sftp.open(posixpath.join(remote_dir, "result.json")) as f:
                return json.loads(f.read().decode())
        except Exception as e:
            return {"status": "error", "config": _echo(config),
                    "error_detail": f"probe produced no result.json (remote exit {rc}): {e}"}
    finally:
        client.close()
