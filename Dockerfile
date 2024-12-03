FROM python:3.11.10

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY . /app

ENTRYPOINT  uvicorn langsecure.server:app --host 0.0.0.0 --reload