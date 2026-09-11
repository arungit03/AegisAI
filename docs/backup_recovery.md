# AegisAI Backup and Recovery Procedures

## Overview

This document describes backup strategies and recovery procedures for the AegisAI system. The system consists of two persistent data stores:

- **PostgreSQL**: Stores user accounts, document metadata, audit logs, and RBAC configuration
- **Qdrant**: Stores vector embeddings for document retrieval

Both must be backed up and recovered together to maintain system integrity.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Docker Compose                        │
├─────────────────────────────────────────────────────────┤
│  postgres  │  qdrant   │  ollama  │  backend  │  nginx   │
│  :5432     │  :6333    │  :11434  │  :8000    │  :80     │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### Docker CLI
All backup and restore operations require Docker CLI access:
```bash
docker ps  # Should list running containers
```

### Backup Storage
Backups should be stored on a separate volume or remote server. Configure:
```bash
BACKUP_DIR=/var/backups/aegisai
```

### Encryption (Recommended)
```bash
# Install gpg for encrypted backups
apt-get install gnupg  # Debian/Ubuntu
```

## Backup Procedures

### 1. PostgreSQL Database Backup

#### Full Backup (Recommended - Run Daily)
```bash
#!/bin/bash
# backup-postgres.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=${BACKUP_DIR:-/var/backups/aegisai}
PGPASSWORD=${POSTGRES_PASSWORD:-aegisai}
CONTAINER_NAME=${POSTGRES_CONTAINER:-aegisai-postgres}

mkdir -p "$BACKUP_DIR/postgres"

docker exec -i "$CONTAINER_NAME" \
  pg_dump -U "${POSTGRES_USER:-aegisai}" "${POSTGRES_DB:-aegisai}" \
  > "$BACKUP_DIR/postgres/aegisai_db_${DATE}.sql"

# Compress the backup
gzip "$BACKUP_DIR/postgres/aegisai_db_${DATE}.sql"

echo "PostgreSQL backup completed: aegisai_db_${DATE}.sql.gz"
```

#### Encrypted Backup
```bash
# After creating the backup, encrypt it
gpg --encrypt --recipient backup@aegisai.com \
  "$BACKUP_DIR/postgres/aegisai_db_${DATE}.sql.gz"

# Verify encrypted backup
gpg --list-packets "$BACKUP_DIR/postgres/aegisai_db_${DATE}.sql.gz.gpg"
```

#### Transaction Log Backup (Point-in-Time Recovery)
For PostgreSQL 16+, enable WAL archiving in `postgresql.conf`:
```conf
wal_level = replica
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'
```

### 2. Qdrant Vector Database Backup

#### Snapshot Method (Recommended)
```bash
#!/bin/bash
# backup-qdrant.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=${BACKUP_DIR:-/var/backups/aegisai}
QDRANT_CONTAINER=${QDRANT_CONTAINER:-aegisai-qdrant}
COLLECTION=${QDRANT_COLLECTION:-aegisai_documents}

mkdir -p "$BACKUP_DIR/qdrant"

# Create snapshot via API
curl -X POST "http://localhost:6333/snapshots/$COLLECTION" \
  -H "Content-Type: application/json" \
  -d '{"name": "aegisai_backup_${DATE}", "wait: true"}'

# Download snapshot
curl -X GET "http://localhost:6333/snapshots/$COLLECTION/aegisai_backup_${DATE}" \
  -o "$BACKUP_DIR/qdrant/aegisai_vectors_${DATE}.snapshot"

echo "Qdrant backup completed: aegisai_vectors_${DATE}.snapshot"
```

#### Volume Backup Method
```bash
# Stop container first (brief downtime)
docker stop "$QDRANT_CONTAINER"

# Create volume snapshot
docker run --rm \
  -v aegisai_qdrant_data:/source \
  -v "$BACKUP_DIR/qdrant":/backup \
  alpine tar czf "/backup/qdrant_volume_${DATE}.tar.gz" /source

# Restart container
docker start "$QDRANT_CONTAINER"
```

### 3. Combined Backup Script

```bash
#!/bin/bash
# backup-all.sh - Complete backup of AegisAI system

set -euo pipefail

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR=${BACKUP_DIR:-/var/backups/aegisai}
RETENTION_DAYS=${RETENTION_DAYS:-7}

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR/postgres" "$BACKUP_DIR/qdrant"

echo "=== Starting AegisAI backup: $DATE ==="

# 1. PostgreSQL backup
echo "[1/3] Backing up PostgreSQL..."
BACKUP_FILE="$BACKUP_DIR/postgres/aegisai_db_${DATE}.sql"
docker exec -i "${POSTGRES_CONTAINER:-aegisai-postgres}" \
  pg_dump -U "${POSTGRES_USER:-aegisai}" "${POSTGRES_DB:-aegisai}" \
  > "$BACKUP_FILE"
gzip "$BACKUP_FILE"

# 2. Qdrant snapshot backup
echo "[2/3] Backing up Qdrant..."
curl -X POST "http://localhost:6333/snapshots/${QDRANT_COLLECTION:-aegisai_documents}" \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"aegisai_backup_${DATE}\", \"wait\": true}" >/dev/null 2>&1

curl -X GET "http://localhost:6333/snapshots/${QDRANT_COLLECTION:-aegisai_documents}/aegisai_backup_${DATE}" \
  -o "$BACKUP_DIR/qdrant/aegisai_vectors_${DATE}.snapshot"

# 3. Configuration backup
echo "[3/3] Backing up configuration..."
cp .env.production.example "$BACKUP_DIR/aegisai_config_${DATE}.env.example"

# 4. Cleanup old backups
echo "Cleaning up backups older than ${RETENTION_DAYS} days..."
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*.snapshot" -mtime +$RETENTION_DAYS -delete

echo "=== Backup completed successfully ==="
ls -lh "$BACKUP_DIR/postgres/aegisai_db_${DATE}.sql.gz"
ls -lh "$BACKUP_DIR/qdrant/aegisai_vectors_${DATE}.snapshot"
ls -lh "$BACKUP_DIR/aegisai_config_${DATE}.env.example"
```

### 4. Automated Backups with Cron

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /opt/aegisai/scripts/backup-all.sh >> /var/log/aegisai/backup.log 2>&1

# Weekly full volume backup on Sundays at 3 AM
0 3 * * 0 /opt/aegisai/scripts/backup-all.sh --full >> /var/log/aegisai/backup.log 2>&1
```

## Recovery Procedures

### 1. PostgreSQL Database Restore

#### Full Restore
```bash
#!/bin/bash
# restore-postgres.sh

DATE=$1  # e.g., 20250902_020000
BACKUP_DIR=${BACKUP_DIR:-/var/backups/aegisai}
PGPASSWORD=${POSTGRES_PASSWORD:-aegisai}
CONTAINER_NAME=${POSTGRES_CONTAINER:-aegisai-postgres}

BACKUP_FILE="$BACKUP_DIR/postgres/aegisai_db_${DATE}.sql.gz"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Create new database (drop existing)
docker exec -i "$CONTAINER_NAME" \
  psql -U "${POSTGRES_USER:-aegisai}" -d postgres -c "DROP DATABASE IF EXISTS ${POSTGRES_DB:-aegisai};"
docker exec -i "$CONTAINER_NAME" \
  psql -U "${POSTGRES_USER:-aegisai}" -d postgres -c "CREATE DATABASE ${POSTGRES_DB:-aegisai};"

# Restore
gunzip -c "$BACKUP_FILE" | \
  docker exec -i "$CONTAINER_NAME" \
  psql -U "${POSTGRES_USER:-aegisai}" -d "${POSTGRES_DB:-aegisai}"

echo "PostgreSQL restore completed from: $BACKUP_FILE"
```

#### Encrypted Restore
```bash
# Decrypt first
gpg --decrypt "$BACKUP_DIR/postgres/aegisai_db_${DATE}.sql.gz.gpg" \
  | gunzip > /tmp/restore.sql

# Then restore
cat /tmp/restore.sql | docker exec -i "$CONTAINER_NAME" \
  psql -U "${POSTGRES_USER:-aegisai}" -d "${POSTGRES_DB:-aegisai}"
```

### 2. Qdrant Vector Database Restore

#### Restore from Snapshot
```bash
#!/bin/bash
# restore-qdrant.sh

DATE=$1  # e.g., 20250902_020000
BACKUP_DIR=${BACKUP_DIR:-/var/backups/aegisai}
QDRANT_CONTAINER=${QDRANT_CONTAINER:-aegisai-qdrant}
COLLECTION=${QDRANT_COLLECTION:-aegisai_documents}

SNAPSHOT_FILE="$BACKUP_DIR/qdrant/aegisai_vectors_${DATE}.snapshot"

if [ ! -f "$SNAPSHOT_FILE" ]; then
    echo "Snapshot file not found: $SNAPSHOT_FILE"
    exit 1
fi

# Upload snapshot
curl -X PUT "http://localhost:6333/snapshots/$COLLECTION" \
  -H "Content-Type: application/octet-stream" \
  --data-binary "@$SNAPSHOT_FILE"

# Restore collection from snapshot
curl -X POST "http://localhost:6333/collections/$COLLECTION/snapshots/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@$SNAPSHOT_FILE" \
  -F "name=aegisai_backup_${DATE}"

echo "Qdrant restore completed from: $SNAPSHOT_FILE"
```

#### Volume Restore
```bash
# Stop containers
docker-compose -f docker-compose.yml down

# Restore volume
docker run --rm \
  -v aegisai_qdrant_data:/target \
  -v "$BACKUP_DIR/qdrant":/backup \
  alpine \
  sh -c "rm -rf /target/* && tar xzf /backup/qdrant_volume_${DATE}.tar.gz -C /target --strip=1"

# Start containers
docker-compose -f docker-compose.yml up -d
```

### 3. Disaster Recovery (Complete System Restore)

```bash
#!/bin/bash
# disaster-recovery.sh

BACKUP_DATE=$1
BACKUP_DIR=${BACKUP_DIR:-/var/backups/aegisai}

echo "=== Starting disaster recovery from backup: $BACKUP_DATE ==="

# 1. Stop all services
echo "[1/5] Stopping services..."
docker-compose -f docker-compose.yml down

# 2. Restore PostgreSQL
echo "[2/5] Restoring PostgreSQL database..."
./restore-postgres.sh "$BACKUP_DATE"

# 3. Restore Qdrant
echo "[3/5] Restoring Qdrant vector database..."
./restore-qdrant.sh "$BACKUP_DATE"

# 4. Restore configuration
echo "[4/5] Restoring configuration..."
cp "$BACKUP_DIR/aegisai_config_${BACKUP_DATE}.env.example" .env.production

# 5. Start services
echo "[5/5] Starting services..."
docker-compose -f docker-compose.yml up -d

# Wait for health checks
sleep 30

echo "=== Disaster recovery completed ==="
echo "Verify system at: http://localhost:8000/api/health"
curl -s http://localhost:8000/api/health/ready | python -m json.tool
```

## Backup Verification

### Test Database Backup Integrity
```bash
# Verify PostgreSQL backup
gunzip -t "$BACKUP_DIR/postgres/aegisai_db_${DATE}.sql.gz"
if [ $? -eq 0 ]; then
    echo "Backup file integrity: OK"
else
    echo "Backup file corrupted!"
    exit 1
fi

# Verify backup contains expected tables
gunzip -c "$BACKUP_DIR/postgres/aegisai_db_${DATE}.sql.gz" | grep -q "CREATE TABLE.*users"
gunzip -c "$BACKUP_DIR/postgres/aegisai_db_${DATE}.sql.gz" | grep -q "CREATE TABLE.*documents"
```

### Test Qdrant Snapshot Integrity
```bash
# Verify snapshot file exists and is non-empty
if [ -s "$BACKUP_DIR/qdrant/aegisai_vectors_${DATE}.snapshot" ]; then
    echo "Snapshot integrity: OK"
else
    echo "Snapshot file missing or empty!"
    exit 1
fi

# Verify collection info via API
curl -s "http://localhost:6333/collections/${QDRANT_COLLECTION:-aegisai_documents}" | python -m json.tool
```

## Retention Policy

| Backup Type | Retention | Schedule | Storage Location |
|-------------|-----------|----------|------------------|
| Full PostgreSQL dump | 7 days | Daily at 02:00 | Local backup volume |
| Qdrant snapshot | 7 days | Daily at 02:05 | Local backup volume |
| Configuration | 30 days | Daily | Local backup volume |
| Weekly volume backup | 30 days | Sunday at 03:00 | Remote backup server |
| Point-in-time WAL | 24 hours | Continuous | Local WAL archive |

## Point-in-Time Recovery (PITR)

### Enable WAL Archiving
```conf
# In PostgreSQL configuration
wal_level = replica
archive_mode = on
archive_command = 'cp %p /var/lib/postgresql/wal_archive/%f'
```

### Restore to Specific Point in Time
```bash
# 1. Restore base backup
gunzip -c base_backup.sql.gz | psql -U aegisai -d aegisai

# 2. Create recovery configuration
cat > /var/lib/postgresql/data/recovery.conf <<EOF
restore_command = 'cp /var/lib/postgresql/wal_archive/%f %p'
recovery_target_time = '2025-09-02 14:30:00'
recovery_target_timeline = 'latest'
EOF

# 3. Start PostgreSQL to apply WAL logs
docker-compose restart postgres
```

## Testing Backups

### Monthly Restore Test
```bash
#!/bin/bash
# test-backup.sh - Monthly restore validation

echo "=== Testing backup integrity ==="

# 1. Create temporary restore environment
docker-compose -f docker-compose.test.yml up -d

# 2. Restore latest backup
DATE=$(date +%Y%m%d)
./restore-postgres.sh "$DATE"
./restore-qdrant.sh "$DATE"

# 3. Verify restored data
docker exec -i aegisai-postgres-test \
  psql -U aegisai -d aegisai -c "SELECT COUNT(*) FROM documents;" \
  | grep -q "[1-9]" && echo "Documents: OK" || echo "Documents: FAILED"

# 4. Verify vector search works
curl -s -X POST "http://localhost:8000/api/chat/query" \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{"question":"test"}' | python -m json.tool

# 5. Clean up
docker-compose -f docker-compose.test.yml down -v

echo "=== Backup test completed ==="
```

## Security Considerations

1. **Encryption at Rest**: All backup files should be encrypted before storage
2. **Access Control**: Backup directories should have restricted access (700 permissions)
3. **Secret Rotation**: Backup scripts should use environment variables for credentials, not hardcoded values
4. **Offsite Storage**: Store encrypted backups on a separate server or cloud storage bucket
5. **Audit Trail**: Log all backup and restore operations for compliance

## Emergency Procedures

### If Primary Database is Corrupted
1. Immediately stop all write operations
2. Switch to the most recent backup
3. Apply WAL logs if available
4. Notify system administrators

### If Qdrant Vectors are Lost
1. Vectors can be re-generated from document files
2. Use `process_document_upload` API to re-ingest all documents
3. The ingestion pipeline will regenerate embeddings automatically
4. PostgreSQL metadata remains intact for document tracking

## Troubleshooting

### Common Issues

**Backup fails with connection refused:**
```bash
# Check if containers are running
docker-compose ps
# Ensure PostgreSQL is healthy
docker exec aegisai-postgres pg_isready -U aegisai
```

**Restore fails with permission denied:**
```bash
# Ensure correct file ownership
chown $(id -u):$(id -g) "$BACKUP_DIR/..."/*.gz
```

**Qdrant snapshot upload fails:**
```bash
# Check Qdrant version matches snapshot version
curl -s "http://localhost:6333/" | python -m json.tool
```

---

**Next Review Date**: Monthly  
**Owner**: DevOps Team  
**Last Updated**: 2025-09-02