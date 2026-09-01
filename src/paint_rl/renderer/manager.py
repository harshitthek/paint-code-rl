"""Renderer service manager: lifecycle, health checks, connection pooling, and process isolation.
"""
import os
import sys
import time
import signal
import atexit
import subprocess
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


class RendererService:
    """Manages the Node.js WebGL renderer daemon process and client communication."""

    def __init__(self, host: str = "127.0.0.1", port: int = 3000, timeout: int = 15, auto_spawn: bool = True):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.auto_spawn = auto_spawn
        self.base_url = f"http://{host}:{port}"
        self._proc = None
        
        # Persistent connection pool for high-throughput RL rollouts
        self._session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.2,
            status_forcelist=[502, 503, 504],
        )
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retries)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def is_healthy(self) -> bool:
        """Check if the renderer server is responding to health checks."""
        for url in [self.base_url, f"http://localhost:{self.port}"]:
            try:
                r = self._session.get(f"{url}/health", timeout=1.5)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    self.base_url = url
                    return True
            except Exception:
                pass
        return False

    def ensure_started(self, max_wait_sec: int = 15) -> bool:
        """Ensure the Node.js WebGL renderer daemon is running and responsive."""
        if self.is_healthy():
            return True
            
        if not self.auto_spawn:
            return False

        # Locate renderer directory by finding repo root
        curr = os.path.dirname(os.path.abspath(__file__))
        repo_root = None
        for _ in range(6):
            if os.path.exists(os.path.join(curr, "pyproject.toml")):
                repo_root = curr
                break
            parent = os.path.dirname(curr)
            if parent == curr:
                break
            curr = parent
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        
        renderer_dir = os.path.join(repo_root, "renderer")
        server_js = os.path.join(renderer_dir, "server.js")
        
        if not os.path.exists(server_js):
            print(f"[WARN] Cannot find renderer server script at {server_js}")
            return False

        print(f"Starting Node.js WebGL renderer daemon on port {self.port}...")
        env = os.environ.copy()
        env["PORT"] = str(self.port)
        
        try:
            # Create a dedicated process group on POSIX systems to allow clean subtree termination
            kwargs = {}
            if hasattr(os, "setsid"):
                kwargs["preexec_fn"] = os.setsid
                
            # DEVNULL prevents OS pipe buffer exhaustion deadlock during long runs
            self._proc = subprocess.Popen(
                ["node", "server.js"],
                cwd=renderer_dir,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                **kwargs
            )
            atexit.register(self.shutdown)
        except Exception as e:
            print(f"[ERROR] Failed to spawn node process: {e}")
            return False

        # Wait for health check
        start_t = time.time()
        while time.time() - start_t < max_wait_sec:
            if self._proc.poll() is not None:
                _, stderr = self._proc.communicate()
                print(f"[ERROR] Node.js renderer process exited with code {self._proc.returncode}!")
                if stderr:
                    print(f"Stderr: {stderr}")
                return False
                
            if self.is_healthy():
                print(f"✅ Renderer service is ready at {self.base_url}")
                return True
            time.sleep(0.4)

        print("[WARN] Renderer service did not become healthy within timeout.")
        return False

    def render(self, code: str, seed: int = 42, prompt: str = "") -> dict:
        """Render generative p5.js/p5.brush code and capture canvas PNG."""
        self.ensure_started()
        try:
            resp = self._session.post(
                f"{self.base_url}/render",
                json={"code": code, "seed": seed, "prompt": prompt},
                timeout=self.timeout
            )
            if resp.status_code == 429:
                return {"success": False, "error_classification": "RENDERER_OVERLOAD", "runtime_error": "Backpressure applied"}
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            # Attempt one automatic restart if connection was dropped
            if self.ensure_started(max_wait_sec=5):
                try:
                    resp = self._session.post(
                        f"{self.base_url}/render",
                        json={"code": code, "seed": seed, "prompt": prompt},
                        timeout=self.timeout
                    )
                    resp.raise_for_status()
                    return resp.json()
                except Exception as e:
                    return {"success": False, "error_classification": "RENDERER_UNAVAILABLE", "runtime_error": str(e)}
            return {"success": False, "error_classification": "RENDERER_UNAVAILABLE", "runtime_error": "Cannot connect to renderer server"}
        except Exception as e:
            return {"success": False, "error_classification": "RENDERER_HTTP_ERROR", "runtime_error": str(e)}

    def shutdown(self):
        """Cleanly terminate the renderer process and all child Chrome instances."""
        if self._proc is not None:
            try:
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    try:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    try:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    self._proc.kill()
            self._proc = None
