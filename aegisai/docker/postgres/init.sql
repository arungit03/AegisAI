-- PostgreSQL Initialization Script for AegisAI
-- This runs automatically on first database creation

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create enum types
DO $$ BEGIN
    CREATE TYPE user_role AS ENUM ('admin', 'manager', 'engineer', 'employee');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE document_classification AS ENUM ('PUBLIC_INTERNAL', 'CONFIDENTIAL', 'RESTRICTED', 'HIGHLY_RESTRICTED');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE document_status AS ENUM ('uploading', 'processing', 'ready', 'failed', 'archived');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE permission_type AS ENUM ('read', 'write', 'admin');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE audit_action AS ENUM (
        'login', 'logout', 'login_failed', 'password_change',
        'user_create', 'user_update', 'user_delete',
        'role_create', 'role_update', 'role_delete',
        'department_create', 'department_update', 'department_delete',
        'document_upload', 'document_update', 'document_delete', 'document_download',
        'document_reindex', 'document_permission_grant', 'document_permission_revoke',
        'chat_message', 'conversation_create', 'conversation_delete',
        'settings_update', 'system_config_change'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Set timezone
SET timezone = 'UTC';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE aegisai TO aegisai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aegisai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aegisai;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO aegisai;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO aegisai;