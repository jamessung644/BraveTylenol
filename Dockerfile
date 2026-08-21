FROM python:3.13.15-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/appuser

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt \
    && useradd --create-home --uid 10001 --user-group appuser

COPY --chown=appuser:appuser app.py main.py ./
COPY --chown=appuser:appuser lunit_hackathon ./lunit_hackathon

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import http.client; c=http.client.HTTPConnection('127.0.0.1', 8000, timeout=3); c.request('GET', '/health'); r=c.getresponse(); raise SystemExit(0 if r.status == 200 else 1)"

CMD ["python", "main.py", "serve"]
