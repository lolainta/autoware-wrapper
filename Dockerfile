FROM ghcr.io/autowarefoundation/autoware:universe-sensing-perception-devel-cuda-20260209

ENV DEBIAN_FRONTEND=noninteractive

RUN <<EOF
    apt update
    apt install -y \
        git \
        ca-certificates \
        xserver-xorg \
        libvulkan1 \
        libsdl2-2.0-0 \
        libomp5 \
        xdg-user-dirs
    rm -rf /var/lib/apt/lists/*
EOF

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=docker.io/tonychi/carla:0.9.16 /opt/carla/ /opt/carla

WORKDIR /app
COPY --chown=carla:carla ./pyproject.toml .
COPY --chown=carla:carla ./uv.lock .
RUN uv sync --locked
RUN uv add /opt/carla/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
ENV PYTHONPATH=/opt/carla/PythonAPI/carla/
COPY . .

ENV PORT=50051
ENV CARLA_PORT=2000

ENTRYPOINT [ "/bin/bash" ]
CMD [ "/app/entrypoint.sh" ]
