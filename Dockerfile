# CPU-only toolchain for simulation and analysis: Rust stable, uv, Python 3.14.
# The repository is mounted at /work at run time, not copied in.
FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo UV_INSTALL_DIR=/opt/uv \
    UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy
ENV PATH=/opt/cargo/bin:/opt/uv:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
        --component rustfmt,clippy --no-modify-path \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv python install 3.14

WORKDIR /work
CMD ["bash"]
