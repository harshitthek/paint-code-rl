import sqlite3
import os
import json
import hashlib

class CacheManager:
    def __init__(self, db_path: str = "artifacts/cache.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_db()
        
    def _init_db(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS vlm_cache
                     (cache_key TEXT PRIMARY KEY, 
                      decision TEXT, 
                      raw_response TEXT, 
                      latency REAL,
                      model TEXT,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        self.conn.commit()

    def generate_vlm_key(self, cand_hash: str, ref_hash: str, prompt: str, model: str, prompt_version: str) -> str:
        key_str = f"{cand_hash}|{ref_hash}|{prompt}|{model}|{prompt_version}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get_vlm(self, cache_key: str):
        c = self.conn.cursor()
        c.execute("SELECT decision, raw_response, latency, model FROM vlm_cache WHERE cache_key=?", (cache_key,))
        row = c.fetchone()
        if row:
            return {
                "decision": row[0],
                "raw_response": json.loads(row[1]),
                "latency": row[2],
                "model": row[3],
                "cache_hit": True
            }
        return None

    def set_vlm(self, cache_key: str, decision: str, raw_response: dict, latency: float, model: str):
        c = self.conn.cursor()
        c.execute('''INSERT OR REPLACE INTO vlm_cache 
                     (cache_key, decision, raw_response, latency, model) 
                     VALUES (?, ?, ?, ?, ?)''',
                  (cache_key, decision, json.dumps(raw_response), latency, model))
        self.conn.commit()
