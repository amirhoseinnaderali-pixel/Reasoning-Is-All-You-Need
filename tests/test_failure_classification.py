from cttr_vps.docker_sandbox import DockerSandbox
def test_digest_is_required():
    try:DockerSandbox("gcc:latest")
    except RuntimeError:return
    assert False
