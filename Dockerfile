FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SUPPORT_AGENT_REPORTS=/app/reports

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY reports ./reports
RUN pip install . && useradd --system --uid 10001 agent

USER 10001
EXPOSE 8090
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/api/status', timeout=2)" || exit 1

ENTRYPOINT ["support-agent"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8090"]
