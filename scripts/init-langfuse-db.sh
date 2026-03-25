#!/bin/bash
# Crée l'utilisateur et la database langfuse dans PostgreSQL si absents
# Exécuté manuellement une fois : docker compose exec postgres bash /app/scripts/init-langfuse-db.sh

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'langfuse') THEN
            CREATE USER langfuse WITH PASSWORD 'langfuse';
        END IF;
    END
    \$\$;

    SELECT 'CREATE DATABASE langfuse OWNER langfuse'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec

    GRANT ALL PRIVILEGES ON DATABASE langfuse TO langfuse;
EOSQL

echo "Langfuse DB initialized."
