"""Renderer service manager: auto-start, health-check, lifecycle management."""
import os
import subprocess
import time
import requests
import atexit
import signal


class RendererService:
    """Manages the Node.js/Puppeteer WebGL rendering service subprocess."""
    
    def __init__(self, port: int = 3000, timeout: int = 25):
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._proc = None
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=2)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    def is_healthy(self) -> bool:
        """Check if the renderer service is up and responding."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status") == "ok"
            return False
        except Exception:
            return False

    def ensure_started(self, max_wait_sec: int = 15) -> bool:
        """Ensure the Node.js renderer service is running and healthy."""
        if self.is_healthy():
            return True

        if self._proc is not None:
            if self._proc.poll() is None:
                # Running but not healthy yet, wait a bit
                start_t = time.time()
                while time.time() - start_t < 3:
                    if self.is_healthy():
                        return True
                    time.sleep(0.3)
                self.shutdown()
            else:
                self._proc = None

        # Discover renderer dir relative to project root
        repo_root = None
        curr = os.path.abspath(os.path.dirname(__file__))
        for _ in range(5):
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

            import shutil
            import platform
            cmd = ["node", "server.js"]
            if platform.system() == "Linux" and shutil.which("xvfb-run"):
                cmd = ["xvfb-run", "-a", "node", "server.js"]
                
            log_dir = os.path.join(self._repo_root, "artifacts", "logs")
            os.makedirs(log_dir, exist_ok=True)
            self._log_file_path = os.path.join(log_dir, "renderer.log")
            self._log_file = open(self._log_file_path, "a", encoding="utf-8")

            self._proc = subprocess.Popen(
                cmd,
                cwd=renderer_dir,
                env=env,
                stdout=self._log_file,
                stderr=self._log_file,
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
                print(f"[ERROR] Node.js renderer process exited with code {self._proc.returncode}!")
                try:
                    if os.path.exists(self._log_file_path):
                        with open(self._log_file_path, "r", encoding="utf-8", errors="replace") as lf:
                            lines = lf.readlines()
                            if lines:
                                print("[Renderer Log Output]:")
                                for l in lines[-10:]:
                                    print(f"   {l.rstrip()}")
                except Exception:
                    pass
                self._proc = None
                return False
                
            if self.is_healthy():
                print(f"[OK] Renderer service is ready at {self.base_url}")
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

    def render_batch(self, items, return_base64=False):
        """Render multiple code completions concurrently via the batch endpoint.
        
        Args:
            items: List of dicts with keys 'code', 'seed', and optionally 'prompt'.
            return_base64: If True, returns in-memory base64 PNG instead of disk paths.
            
        Returns:
            List of render result dicts (one per item), ordered by batch_index.
        """
        self.ensure_started()
        try:
            resp = self._session.post(
                f"{self.base_url}/render_batch",
                json={"items": items, "return_base64": return_base64},
                timeout=self.timeout * 2  # Batch may take longer
            )
            if resp.status_code == 429:
                return [{"success": False, "error_classification": "RENDERER_OVERLOAD", "runtime_error": "Too many concurrent requests (HTTP 429)"} for _ in items]
            resp.raise_for_status()
            data = resp.json()
            if data.get("success") and "results" in data:
                return sorted(data["results"], key=lambda r: r.get("batch_index", 0))
            return [{"success": False, "error_classification": "BATCH_ERROR"} for _ in items]
        except Exception as e:
            return [{"success": False, "error_classification": "BATCH_HTTP_ERROR", "runtime_error": str(e)} for _ in items]

    def shutdown(self):
        """Cleanly terminate the renderer process and all child Chrome instances."""
        if self._proc is not None:
            if self._proc.poll() is not None:
                self._proc = None
                return
            try:
                if hasattr(os, "killpg") and hasattr(os, "getpgid") and self._proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                if self._proc is not None and self._proc.poll() is None:
                    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                        try:
                            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    else:
                        self._proc.kill()
            self._proc = None
