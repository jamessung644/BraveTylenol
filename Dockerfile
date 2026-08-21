FROM python:3.13.15-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY main.py /app/main.py

USER 65532:65532

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=2s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]

CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
