"""SSH session owner: terminate only this node's torchrun process group."""
import os
import signal
import subprocess
import sys


def main():
    child = None
    interrupted = [False]

    def stop(signum, frame):
        interrupted[0] = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
        signal.signal(sig, stop)
    child = subprocess.Popen(sys.argv[1:], start_new_session=True)
    try:
        while child.poll() is None:
            try:
                return child.wait(timeout=1 if not interrupted[0] else 30)
            except subprocess.TimeoutExpired:
                if interrupted[0]:
                    os.killpg(child.pid, signal.SIGKILL)
                    return child.wait()
        return child.returncode
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    sys.exit(main())
