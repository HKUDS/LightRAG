# Installing and Using PostgreSQL with LightRAG

This guide provides step-by-step instructions on setting up PostgreSQL for use with LightRAG, a tool designed to enhance large language model (LLM) performance using retrieval-augmented generation techniques.

## Prerequisites

Before beginning this setup, ensure that you have administrative access to your server or local machine and can install software packages.

### 1. Install PostgreSQL

First, update your package list and install PostgreSQL:

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

Start the PostgreSQL service if it isn’t already running:

```bash
sudo systemctl start postgresql
```

Ensure that PostgreSQL starts on boot:

```bash
sudo systemctl enable postgresql
```

### 2. Set a Password for Your Postgres Role

By default, PostgreSQL creates a user named `postgres`. You'll need to set a password for this role or create another role with a password.

To set a password for the `postgres` user:

```bash
sudo -u postgres psql
```

Inside the PostgreSQL shell, run:

```sql
ALTER USER postgres WITH PASSWORD 'your_secure_password';
\q
```

Alternatively, to create a new role with a password:

```bash
sudo -u postgres createuser --interactive
```

You'll be prompted for the name of the new role and whether it should have superuser permissions. Then set a password:

```sql
ALTER USER your_new_role WITH PASSWORD 'your_secure_password';
\q
```

### 3. Install PGVector and Age Extensions

Install PGVector:
```bash
sudo apt install postgresql-server-dev-all
cd /tmp
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```
Install age:
```bash
sudo apt-get install build-essential libpq-dev
cd /tmp
git clone https://github.com/apache/age.git
cd age
make
sudo make install
```

### 4. Create a Database for LightRAG

Create an empty database to store your data:

```bash
sudo -u postgres createdb your_database
```

### 5. Activate PGVector Extension in the Database

Switch to the newly created database and enable the `pgvector` extension:

```bash
sudo -u postgres psql -d your_database
```

Inside the PostgreSQL shell, run:

```sql
CREATE EXTENSION vector;
```

Verify installation by checking the extension version within this specific database:

```sql
SELECT extversion FROM pg_extension WHERE extname = 'vector';
\q
```

### 6. Install LightRAG with API Access

Install LightRAG using pip, targeting the API package for server-side use:

```bash
pip install "lightrag-hku[api]"
```

### 7. Configure `config.ini`

Create a configuration file to specify PostgreSQL connection details and other settings:

In your project directory, create a `config.ini` file with the following content:

```ini
[postgres]
host = localhost
port = 5432
user = your_role_name
password = your_password
database = your_database
workspace = default
```

Replace placeholders like `your_role_name`, `your_password`, and `your_database` with actual values.

### 8. Run LightRAG Server

Start the LightRAG server using specified options:

```bash
lightrag-server --port 9621 --key sk-somepassword --kv-storage PGKVStorage --graph-storage PGGraphStorage --vector-storage PGVectorStorage --doc-status-storage PGDocStatusStorage
```

Replace the `port` number with your desired port number (default is 9621) and `your-secret-key` with a secure key.

## Conclusion

With PostgreSQL set up to work with LightRAG, you can now leverage vector storage and retrieval-augmented capabilities for enhanced language model operations. Adjust configurations as needed based on your specific environment and use case requirements.
