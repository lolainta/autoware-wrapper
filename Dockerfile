FROM ghcr.io/autowarefoundation/autoware:universe-devel-cuda-20260209

ENV DEBIAN_FRONTEND=noninteractive

RUN <<EOF
    apt update
    apt install -y git
    rm -rf /var/lib/apt/lists/*
EOF

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY --chown=carla:carla ./pyproject.toml .
COPY --chown=carla:carla ./uv.lock .
RUN uv sync --locked
COPY misc/sbsvf.launch.xml /opt/autoware/share/autoware_launch/launch/
COPY . .

RUN pip install pyyaml
COPY misc/config.py /tmp/config.py
RUN python3 /tmp/config.py --apply

ENV PORT=50051

ENTRYPOINT [ "/bin/bash" ]
CMD [ "/app/entrypoint.sh" ]
