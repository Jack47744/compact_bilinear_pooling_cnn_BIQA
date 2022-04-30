FROM python:3.9-slim AS compile-image
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential gcc

WORKDIR /opt/venv

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY ./requirements.txt /opt/venv
RUN pip install -r requirements.txt

COPY . .

FROM python:3.9-slim AS build-image
COPY --from=compile-image /opt/venv /opt/venv

WORKDIR /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8600"]
