FROM python:3.10-slim


# we probably need build tools?
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    python3-dev

WORKDIR /app
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

COPY . .

CMD ["streamlit", "run", "Home.py"]