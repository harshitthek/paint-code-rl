"""Renderer service manager: auto-start, health-check, lifecycle management."""
import os
import subprocess
import time
import requests
import atexit
import signal


import secrets


class RendererService:
    """Manages the Node.js/Puppeteer WebGL rendering service subprocess."""
    
    def __init__(self, port: int = 3000, timeout: int = 60):
        self.port = port
        self.timeout = timeout
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.shutdown_token = secrets.token_hex(16)
        self._proc = None
        self._log_file = None
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

        # Auto-install npm packages if node_modules is missing or wiped
        node_modules_dir = os.path.join(renderer_dir, "node_modules")
        if not os.path.exists(node_modules_dir) or not os.path.exists(os.path.join(node_modules_dir, "express")):
            import shutil
            print("📦 renderer/node_modules missing. Auto-installing npm dependencies...")
            npm_bin = shutil.which("npm") or "/usr/local/bin/npm"
            try:
                subprocess.run([npm_bin, "install", "--no-audit", "--no-fund"], cwd=renderer_dir, check=True)
                print("✅ npm dependencies installed successfully!")
            except Exception as ne:
                print(f"[WARN] Automatic npm install failed: {ne}")

        print(f"Starting Node.js WebGL renderer daemon on port {self.port}...")
        env = os.environ.copy()
        env["PORT"] = str(self.port)
        env["RENDERER_SHUTDOWN_TOKEN"] = self.shutdown_token
        
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
                
            log_dir = os.path.join(repo_root, "artifacts", "logs")
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
        except requests.exceptions.ReadTimeout:
            return {"success": False, "error_classification": "TIMEOUT", "runtime_error": f"Renderer HTTP socket read timed out after {self.timeout}s"}
        except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
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
            # Resilient fallback: render individually if batch fails so reward is never dropped
            try:
                results = []
                for idx, item in enumerate(items):
                    r = self.render(item.get("code", ""), seed=item.get("seed", 42), prompt=item.get("prompt", ""))
                    r["batch_index"] = idx
                    results.append(r)
                    if not r.get("success") and r.get("error_classification") == "RENDERER_UNAVAILABLE":
                        for rem_idx in range(idx + 1, len(items)):
                            results.append({"success": False, "error_classification": "RENDERER_UNAVAILABLE", "batch_index": rem_idx})
                        break
                return results
            except Exception:
                return [{"success": False, "error_classification": "BATCH_HTTP_ERROR", "runtime_error": str(e)} for _ in items]

    def shutdown(self) -> bool:
        """Cleanly terminate the renderer process and all child Chrome instances.
        
        Returns:
            bool: True if shutdown succeeded or server was already down; False if shutdown failed.
        """
        # For externally managed renderers (_proc is None), use the environment token.
        # For self-managed subprocesses (_proc is not None), use self.shutdown_token.
        token = os.environ.get("RENDERER_SHUTDOWN_TOKEN", "") if self._proc is None else getattr(self, "shutdown_token", "")
        if not token:
            token = getattr(self, "shutdown_token", "") or os.environ.get("RENDERER_SHUTDOWN_TOKEN", "")

        shutdown_successful = True
        try:
            headers = {"X-Renderer-Token": token}
            resp = self._session.post(
                f"{self.base_url}/shutdown",
                headers=headers,
                timeout=1.0,
                allow_redirects=False
            )
            if resp.status_code == 401:
                print("[ERROR] Renderer shutdown rejected: 401 Unauthorized (invalid X-Renderer-Token)")
                shutdown_successful = False
            elif resp.status_code != 200:
                shutdown_successful = False
        except requests.exceptions.ConnectionError:
            # Server is not running, already shut down
            pass
        except Exception:
            if self.is_healthy():
                shutdown_successful = False

        if hasattr(self, "_log_file") and self._log_file and not self._log_file.closed:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

        if self._proc is not None:
            if self._proc.poll() is not None:
                self._proc = None
                return True
            try:
                if hasattr(os, "killpg") and hasattr(os, "getpgid") and self._proc.poll() is None:
                    try:
                        os.killpg(os.path.getpgid(self._proc.pid) if hasattr(os.path, "getpgid") else os.getpgid(self._proc.pid), signal.SIGTERM)
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

        # Verify externally managed service has actually stopped
        if shutdown_successful and self.is_healthy():
            start_t = time.time()
            while time.time() - start_t < 2.0:
                if not self.is_healthy():
                    break
                time.sleep(0.2)
            if self.is_healthy():
                shutdown_successful = False

        return shutdown_successful

    def restart(self, max_wait_sec: int = 20) -> bool:
        """Force restart the renderer service ensuring fresh code and configuration."""
        shutdown_ok = self.shutdown()
        if not shutdown_ok and self._proc is None and self.is_healthy():
            print("[ERROR] Cannot restart externally managed renderer: shutdown request failed (server remains alive).")
            return False
        time.sleep(1)
        return self.ensure_started(max_wait_sec=max_wait_sec)
