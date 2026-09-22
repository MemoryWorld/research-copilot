from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import uuid


class SessionConflict(ValueError):
    pass


class Store:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, pages INTEGER NOT NULL,
                    chunks INTEGER NOT NULL, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    payload TEXT NOT NULL, vector TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, revision INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY, session_id TEXT REFERENCES sessions(id),
                    role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}');
                CREATE TABLE IF NOT EXISTS traces (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            """)
            if "metadata" not in {row["name"] for row in conn.execute("PRAGMA table_info(messages)")}:
                conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def bind_embedding(self, fingerprint: str):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT value FROM metadata WHERE key='embedding'").fetchone()
            if row and row["value"] != fingerprint:
                raise ValueError("Embedding 配置与数据库索引不一致；请使用独立数据库重新入库")
            conn.execute("INSERT OR IGNORE INTO metadata VALUES ('embedding', ?)", (fingerprint,))

    def add_document(self, document, chunks, vectors, capacity):
        if len(chunks) != len(vectors):
            raise ValueError("Embedding count mismatch")
        document = {**document, "created_at": time.time()}
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if count + len(chunks) > capacity:
                raise ValueError("知识库切片总量超过本地演示上限")
            conn.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                         tuple(document[key] for key in ["id", "name", "pages", "chunks", "created_at"]))
            conn.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?)",
                             [(chunk["id"], document["id"], json.dumps(chunk, ensure_ascii=False), json.dumps(vector))
                              for chunk, vector in zip(chunks, vectors)])
        return document

    def documents(self):
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM documents ORDER BY created_at DESC, id")]

    def delete_document(self, identifier):
        with self.connect() as conn:
            return conn.execute("DELETE FROM documents WHERE id=?", (identifier,)).rowcount > 0

    def corpus(self):
        with self.connect() as conn:
            return [(json.loads(row["payload"]), json.loads(row["vector"]))
                    for row in conn.execute("SELECT payload, vector FROM chunks ORDER BY id")]

    def session(self, identifier=None):
        if identifier is None:
            identifier = uuid.uuid4().hex
            with self.connect() as conn:
                conn.execute("INSERT INTO sessions VALUES (?, 0, ?)", (identifier, time.time()))
        with self.connect() as conn:
            row = conn.execute("SELECT revision FROM sessions WHERE id=?", (identifier,)).fetchone()
            if row is None:
                raise KeyError("对话不存在，请新建对话")
            messages = [{"role": item["role"], "content": item["content"], **json.loads(item["metadata"])}
                        for item in conn.execute(
                "SELECT role, content, metadata FROM messages WHERE session_id=? ORDER BY id", (identifier,))]
        return {"session_id": identifier, "revision": row["revision"], "messages": messages}

    def finish_turn(self, session_id, revision, question, answer, trace, citations=None, refused=False):
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            changed = conn.execute("UPDATE sessions SET revision=revision+1 WHERE id=? AND revision=?",
                                   (session_id, revision)).rowcount
            if not changed:
                raise SessionConflict("同一对话已有其他回答完成，请刷新后重试")
            conn.executemany("INSERT INTO messages(session_id, role, content, metadata) VALUES (?, ?, ?, ?)",
                             [(session_id, "user", question, "{}"), (session_id, "assistant", answer,
                              json.dumps({"citations": citations or [], "refused": refused,
                                          "trace_id": trace["trace_id"]}, ensure_ascii=False))])
            conn.execute("INSERT INTO traces VALUES (?, ?)", (trace["trace_id"], json.dumps(trace, ensure_ascii=False)))

    def save_trace(self, trace):
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO traces VALUES (?, ?)",
                         (trace["trace_id"], json.dumps(trace, ensure_ascii=False)))

    def trace(self, identifier):
        with self.connect() as conn:
            row = conn.execute("SELECT payload FROM traces WHERE id=?", (identifier,)).fetchone()
        return json.loads(row["payload"]) if row else None
