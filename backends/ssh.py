"""SSH backend — run the probe on a rented box / neocloud / your own cluster.

Ships this repo to the box with `rsync`, runs the local flow (capture -> diagnose) there over
`ssh`, and returns the Observation dict. Same contract as `local`; the difference is
only *where* the work happens.

Built on the **system `ssh`/`rsync` binaries**, not paramiko — no new Python dependency, and the
user's existing `~/.ssh/config`, agent, and known_hosts all keep working. Credential resolution,
in order:

1. `AUTOTUNE_SSH_KEY`      -> `ssh -i <key>`
2. `AUTOTUNE_SSH_PASSWORD` -> `sshpass -p <pw> ssh ...` (needs the `sshpass` binary)
3. neither                 -> plain `ssh`, letting its own config/agent resolve the credential

Secrets come from the environment (`.env`), never from `run.toml` — see `backends/README.md`.
`HF_TOKEN` is forwarded to the box so gated models/datasets resolve there.

The remote's stdout is shared with whatever transformers/datasets decide to print, so the
Observation is framed by sentinels and everything outside them is treated as log noise.
"""
import base64
import json
import os
import shlex
import subprocess

_BEGIN = "---AUTOTUNE-OBSERVATION-BEGIN---"
_END = "---AUTOTUNE-OBSERVATION-END---"

_DEFAULT_REMOTE_DIR = "~/autotune"
_DEFAULT_PYTHON = "~/autotune/.venv_at/bin/python"

# rsync/ssh both get these: fail fast rather than hang a probe on a dead box or a host-key prompt.
_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "-o", "StrictHostKeyChecking=accept-new"]


def _settings(config):
    """Pull the ssh block out of the config, with env overrides for the parts that are secret."""
    ssh = config.get("ssh") or {}
    host = ssh.get("host") or os.environ.get("AUTOTUNE_SSH_HOST")
    user = ssh.get("user") or os.environ.get("AUTOTUNE_SSH_USER")
    if not host:
        raise ValueError("ssh backend needs a host: config['ssh']['host'] (from run.toml [ssh])")
    return {
        "target": f"{user}@{host}" if user else host,
        "remote_dir": ssh.get("remote_dir", _DEFAULT_REMOTE_DIR),
        "python": ssh.get("python", _DEFAULT_PYTHON),
        "key": os.environ.get("AUTOTUNE_SSH_KEY") or None,
        "password": os.environ.get("AUTOTUNE_SSH_PASSWORD") or None,
        "sync": ssh.get("sync", True),
    }


def _sh_path(p):
    """Quote a path for the remote shell, keeping a leading `~` expandable.

    `shlex.quote("~/autotune")` yields `'~/autotune'`, which the remote shell takes literally --
    so the default remote_dir would not exist. Route the tilde through `$HOME` instead.
    """
    if p == "~" or p.startswith("~/"):
        return '"$HOME"' + shlex.quote(p[1:]) if len(p) > 1 else '"$HOME"'
    return shlex.quote(p)


def _ssh_cmd(s):
    """The `ssh` invocation prefix, with whichever credential is configured."""
    cmd = []
    if s["password"]:
        cmd += ["sshpass", "-p", s["password"]]
    cmd += ["ssh", *_SSH_OPTS]
    if s["key"]:
        cmd += ["-i", os.path.expanduser(s["key"])]
    return cmd


def _sync(s):
    """Mirror the repo onto the box. Excludes the venv, git, and run outputs — code only."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = ["rsync", "-az", "--delete",
           "--exclude", ".git/", "--exclude", ".venv/", "--exclude", ".venv_at/",
           "--exclude", "__pycache__/", "--exclude", ".pytest_cache/", "--exclude", "runs/",
           "--exclude", ".env", "--exclude", "*.pyc", "--exclude", ".DS_Store",
           "-e", " ".join(shlex.quote(c) for c in _ssh_cmd(s)),
           repo + "/", f"{s['target']}:{s['remote_dir']}/"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rsync to {s['target']} failed:\n{r.stderr.strip()}")


def _push_sample(s, sample_file):
    """Copy the run's probe sample to the same repo-relative path on the box.

    `_sync` excludes `runs/` (outputs, not code), and a sample is too large to ride in the config:
    the config travels as one argv entry, which Linux caps at 128KB. So the file goes separately,
    and the probe's `sample_file` resolves to the same relative path under `remote_dir`.
    """
    if os.path.isabs(sample_file):
        raise ValueError(f"sample_file {sample_file!r} must be relative to the repo root, so it "
                         "resolves to the same path on the box")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local = os.path.join(repo, sample_file)
    if not os.path.isfile(local):
        raise FileNotFoundError(f"sample_file {sample_file!r} not found under {repo}")
    remote_dir = f"{s['remote_dir'].rstrip('/')}/{os.path.dirname(sample_file)}"
    r = subprocess.run(_ssh_cmd(s) + [s["target"], f"mkdir -p {_sh_path(remote_dir)}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"could not create {remote_dir} on {s['target']}:\n{r.stderr.strip()}")
    cmd = ["rsync", "-az", "-e", " ".join(shlex.quote(c) for c in _ssh_cmd(s)),
           local, f"{s['target']}:{s['remote_dir'].rstrip('/')}/{sample_file}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rsync of {sample_file} to {s['target']} failed:\n{r.stderr.strip()}")


def _ship_run_files(s, files):
    """Push script files kept under runs/ (excluded from the code sync): a refactored copy of the
    user's script, per-probe configs. ALWAYS, sync or not: the agent writes a new per-probe config
    for every probe, so with `sync = false` the box would otherwise run without it, or on a stale
    copy. Files outside runs/ ride with the code sync."""
    for f in files:
        if f.startswith("runs/"):
            _push_sample(s, f)


def _check_script(script):
    """A user's training script, or a file it reads (`script_files`), must be repo-relative, so it
    lands at the same path on the box."""
    if os.path.isabs(script):
        raise ValueError(f"script {script!r} must be relative to the repo root, so it resolves "
                         "to the same path on the box (put it in the repo, e.g. under runs/<run>/)")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isfile(os.path.join(repo, script)):
        raise FileNotFoundError(f"script {script!r} not found under {repo}")


# Runs on the box: read the config off argv, run the probe, frame the result in sentinels.
_REMOTE = """
import base64, json, os, sys
sys.path.insert(0, os.path.expanduser({remote_dir!r}))
cfg = json.loads(base64.b64decode(sys.argv[1]).decode())
from backends.local import run as local_run
obs = local_run(cfg)
print({begin!r}); print(json.dumps(obs, default=str)); print({end!r})
"""


def payload_config(config):
    """The config as the box sees it: connection details stripped, device pinned to cuda.

    The `ssh` block never crosses the wire — the box has no use for its own address, and on
    password auth it would otherwise carry the credential into a remote argv.
    """
    return {**{k: v for k, v in config.items() if k not in ("ssh", "env")}, "device": "cuda"}


def run(config):
    """Run one probe on the remote box and return its Observation (a dict)."""
    s = _settings(config)
    shipped = []                                       # bring-your-own-script: the box runs the
    if config.get("script"):                           #   same repo-relative paths after the sync
        shipped = [config["script"], *(config.get("script_files") or [])]
        for f in shipped:
            _check_script(f)
    if s["sync"]:
        _sync(s)
        if config.get("sample_file"):                  # runs/ is excluded from the code sync
            _push_sample(s, config["sample_file"])
    _ship_run_files(s, shipped)

    payload = base64.b64encode(json.dumps(payload_config(config), default=str).encode()).decode()

    script = _REMOTE.format(remote_dir=s["remote_dir"], begin=_BEGIN, end=_END)
    env_prefix = ""
    if os.environ.get("HF_TOKEN"):
        env_prefix = f"HF_TOKEN={shlex.quote(os.environ['HF_TOKEN'])} "
    # config["env"] -> the remote process's environment. Some levers are only settable BEFORE the
    # process starts and so cannot be a runner knob: PYTORCH_CUDA_ALLOC_CONF is read when the CUDA
    # caching allocator initializes, which happens on first allocation, before any runner code runs.
    # Non-secret by construction (it travels in the config, not the env file).
    for k, v in (config.get("env") or {}).items():
        env_prefix += f"{shlex.quote(str(k))}={shlex.quote(str(v))} "
    remote = (f"cd {_sh_path(s['remote_dir'])} && {env_prefix}"
              f"{_sh_path(s['python'])} -c {shlex.quote(script)} {payload}")

    # The box enforces `timeout` itself and returns an error Observation (backends/local.py). This
    # one only guards a dead connection, so it waits longer: the box's answer must win the race.
    r = subprocess.run(_ssh_cmd(s) + [s["target"], remote],
                       capture_output=True, text=True, timeout=config.get("timeout", 1800) + 300)
    out = r.stdout
    if _BEGIN not in out or _END not in out:
        tail = (r.stderr or out)[-2000:]
        raise RuntimeError(f"remote probe returned no Observation (exit {r.returncode}):\n{tail}")
    return json.loads(out.split(_BEGIN, 1)[1].split(_END, 1)[0].strip())
