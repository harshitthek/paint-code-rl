import os
import sys
import time
import atexit
import subprocess
import requests

class RendererService:
    def __init__(self, host: str = "127.0.0.1", port: int = 3000, timeout: int = 15, auto_spawn: bool = True):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.auto_spawn = auto_spawn
        self.base_url = f"http://{host}:{port}"
        self._proc = None

    def is_healthy(self) -> bool:
        for url in [self.base_url, f"http://localhost:{self.port}"]:
            try:
                r = requests.get(f"{url}/health", timeout=1.0)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    self.base_url = url
                    return True
            except Exception:
                pass
        return False

    def ensure_started(self, max_wait_sec: int = 10) -> bool:
        if self.is_healthy():
            return True
            
        if not self.auto_spawn:
            return False

        # Locate renderer directory
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
            self._proc = subprocess.Popen(
                ["node", "server.js"],
                cwd=renderer_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            atexit.register(self.shutdown)
        except Exception as e:
            print(f"[ERROR] Failed to spawn node process: {e}")
            return False

        # Wait for health
        start_t = time.time()
        while time.time() - start_t < max_wait_sec:
            if self._proc.poll() is not None:
                stdout, stderr = self._proc.communicate()
                print(f"[ERROR] Node.js renderer process exited with code {self._proc.returncode}!")
                if stderr:
                    print(f"Stderr: {stderr}")
                return False
                
            if self.is_healthy():
                print(f"✅ Renderer service is ready at {self.base_url}")
                return True
            time.sleep(0.5)

        print("[WARN] Renderer service did not become healthy within timeout.")
        return False

    def render(self, code: str, seed: int = 42, prompt: str = "") -> dict:
        self.ensure_started()
        try:
            resp = requests.post(
                f"{self.base_url}/render",
                json={"code": code, "seed": seed, "prompt": prompt},
                timeout=self.timeout
            )
            if resp.status_code == 429:
                return {"success": False, "error_classification": "RENDERER_OVERLOAD", "runtime_error": "Backpressure applied"}
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            return {"success": False, "error_classification": "RENDERER_UNAVAILABLE", "runtime_error": "Cannot connect to renderer server"}
        except Exception as e:
            return {"success": False, "error_classification": "RENDERER_HTTP_ERROR", "runtime_error": str(e)}

    def shutdown(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None
