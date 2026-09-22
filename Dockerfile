FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 COPILOT_DB=/data/copilot.db COPILOT_MODE=offline
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir . && useradd --uid 10001 --create-home copilot \
    && mkdir /data && chown copilot:copilot /data
USER copilot
EXPOSE 8787
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=2)"
CMD ["python", "-m", "uvicorn", "research_copilot.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8787"]
