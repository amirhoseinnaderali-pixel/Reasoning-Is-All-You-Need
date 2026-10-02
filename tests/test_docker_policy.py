from pathlib import Path
from cttr_vps.docker_sandbox import DockerSandbox

def test_digest_required():
    try:
        DockerSandbox("gcc:latest")
    except RuntimeError:
        return
    assert False

def test_security_flags_are_frozen():
    s=DockerSandbox("gcc:14.2.0-bookworm@sha256:82549aa8f90ada3236a8be70c74543132a76662ef33f0c3271ed802b81584a82")
    cmd=s._build_cmd(Path("/tmp/work"))
    assert ["--network","none"] == cmd[2:4]
    assert "--read-only" in cmd
    assert ["--cap-drop","ALL"] == cmd[cmd.index("--cap-drop"):cmd.index("--cap-drop")+2]
    assert ["--security-opt","no-new-privileges:true"] == cmd[cmd.index("--security-opt"):cmd.index("--security-opt")+2]
    assert "--cpus" in cmd and "--memory" in cmd and "--pids-limit" in cmd and "--tmpfs" in cmd
