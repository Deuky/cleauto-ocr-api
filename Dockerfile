FROM python:3.14 AS base
WORKDIR /app
COPY --from=ghcr.io/deuky/cleauto-ocr/artifact:v1 /artifact ./models/v1
RUN apt-get update && apt-get install -y libgl-dev
CMD ["fastapi", "run", "--host", "0.0.0.0", "--workers", "2", "src/main.py"]

FROM base AS skeleton
COPY requirements.txt Makefile .

FROM skeleton AS build
RUN make build

FROM scratch AS artifact
COPY . /artifact
COPY --from=build /app /artifact

FROM base AS unit
COPY --from=artifact /artifact .
RUN make install