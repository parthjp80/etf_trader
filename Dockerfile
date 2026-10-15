FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY etf_trader/ etf_trader/

# One-shot: the CronJob runs this once per scheduled time and exits
CMD ["python", "-u", "-m", "etf_trader.live"]
