# Start from pgvector image (has vector extension pre-built correctly)
FROM pgvector/pgvector:pg17

# Install build dependencies for AGE
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-17 \
    libreadline-dev \
    zlib1g-dev \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*

# Install Apache AGE 1.6.0 for PG17
RUN cd /tmp \
    && git clone --branch release/PG17/1.6.0 https://github.com/apache/age.git \
    && cd age \
    && make \
    && make install \
    && rm -rf /tmp/age

# Add initialization script to create extensions
RUN echo "CREATE EXTENSION IF NOT EXISTS vector;" > /docker-entrypoint-initdb.d/01-vector.sql \
    && echo "CREATE EXTENSION IF NOT EXISTS age;" > /docker-entrypoint-initdb.d/02-age.sql \
    && echo "SET search_path = ag_catalog, public;" >> /docker-entrypoint-initdb.d/02-age.sql
