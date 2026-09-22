"""Small strict OpenAI-compatible contract; Qwen failures never become offline answers."""
import hashlib
import math
import re

import httpx

from .config import Settings


class ProviderError(RuntimeError):
    pass


def terms(text):
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.casefold())
    # Chinese bigrams preserve useful short concepts without a downloaded tokenizer.
    for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        words.extend(phrase[i:i + 2] for i in range(len(phrase) - 1))
    return words


def normalize(vector):
    if not vector or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vector):
        raise ProviderError("Embedding response contains an invalid vector")
    norm = math.sqrt(sum(v * v for v in vector))
    return [float(v) / norm for v in vector] if norm else [0.0 for v in vector]


class HashEmbedding:
    fingerprint = "offline:feature-hashing:256:v1"

    def embed(self, texts):
        result = []
        for text in texts:
            vector = [0.0] * 256
            for term in terms(text):
                digest = hashlib.sha256(term.encode()).digest()
                vector[int.from_bytes(digest[:4], "big") % 256] += 1 if digest[4] % 2 else -1
            result.append(normalize(vector))
        return result


class QwenClient:
    def __init__(self, settings: Settings, transport=None):
        self.settings = settings
        self.fingerprint = f"qwen:{settings.embedding_model}:1024:v1:{settings.base_url}"
        self.client = httpx.Client(base_url=settings.base_url.rstrip("/") + "/",
                                   headers={"Authorization": f"Bearer {settings.api_key}"},
                                   timeout=httpx.Timeout(45, connect=10), transport=transport,
                                   follow_redirects=False, trust_env=False)

    def close(self):
        self.client.close()

    def _post(self, path, payload):
        try:
            response = self.client.post(path, json=payload)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("non-object response")
            return data
        except (httpx.HTTPError, ValueError) as exc:
            # Never expose Authorization headers or provider response bodies.
            raise ProviderError("Qwen 请求失败，请检查模型权限、区域及服务状态；未降级为离线模式") from exc

    def embed(self, texts):
        vectors = []
        for offset in range(0, len(texts), 10):
            batch = texts[offset:offset + 10]
            data = self._post("embeddings", {"model": self.settings.embedding_model,
                              "input": batch, "dimensions": 1024, "encoding_format": "float"})
            try:
                rows = sorted(data["data"], key=lambda item: item["index"])
                if [row["index"] for row in rows] != list(range(len(batch))):
                    raise ValueError("count/index mismatch")
                values = [normalize(row["embedding"]) for row in rows]
                if any(len(vector) != 1024 for vector in values):
                    raise ValueError("dimension mismatch")
                vectors.extend(values)
            except (KeyError, TypeError, ValueError) as exc:
                raise ProviderError("Qwen embedding response violates the expected contract") from exc
        return vectors

    def complete(self, messages, tools=None, json_mode=False):
        payload = {"model": self.settings.model, "messages": messages, "temperature": 0,
                   "max_tokens": 1500, "stream": False, "enable_thinking": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("chat/completions", payload)
        try:
            message = data["choices"][0]["message"]
            if not isinstance(message, dict):
                raise ValueError("invalid message")
            return message, data.get("usage", {})
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("Qwen chat response violates the expected contract") from exc
