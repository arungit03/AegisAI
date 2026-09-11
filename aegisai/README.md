# AegisAI

A private, secure, enterprise-grade AI assistant built for the Smart India Hackathon. All LLM inference, embeddings, RAG, and document processing run locally — your company data **never leaves your network**.

## Features

- **Complete Privacy**: All AI processing runs locally via Ollama — no external API calls
- **Document Classification**: Four-tier classification (PUBLIC_INTERNAL, CONFIDENTIAL, RESTRICTED, HIGHLY_RESTRICTED)
- **Permission-Aware RAG**: Retrieval filters by user permissions before search, not after
- **Role-Based Access Control**: Admin, Manager, Engineer, Employee roles with granular permissions
- **Audit Logging**: Comprehensive security event tracking
- **Modern Stack**: FastAPI + React 18 + TypeScript + PostgreSQL + Qdrant + Docker
- **Offline-First**: Works completely offline after initial model installation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AegisAI System                            │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React 18 + TypeScript + Vite + Tailwind)            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Dashboard | Chat | Documents | Users | Settings | Audit  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Nginx Reverse Proxy                      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐    │
│  │   Backend   │      │  Qdrant     │      │   Ollama    │    │
│  │  (FastAPI)  │◄────►│  (Vectors)  │      │   (LLM)     │    │
│  └─────────────┘      └─────────────┘      └─────────────┘    │
│         │                                             ▲        │
│         ▼                                             │        │
│  ┌─────────────┐                                       │        │
│  │ PostgreSQL  │                                       │        │
│  │  (Primary)  │                                       │        │
│  └─────────────┘                                       │        │
│                                                         │        │
│  All components run in Docker containers on your network      │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker & Docker Compose (v2.0+)
- 16GB+ RAM (for 7B parameter model)
- NVIDIA GPU recommended (for faster inference)

### 1. Clone and Configure

```bash
git clone <repository-url>
cd aegisai

# Copy environment template
cp .env.example .env

# Edit .env with your secure values
# IMPORTANT: Change SECRET_KEY and POSTGRES_PASSWORD
```

### 2. Start All Services

```bash
# Start all services (first run pulls images and downloads models)
docker-compose up -d

# Or with production profile (includes nginx)
docker-compose --profile production up -d
```

### 3. Pull LLM Models

```bash
# Pull the chat model
docker exec aegisai-ollama ollama pull qwen2.5:7b-instruct

# Pull the embedding model
docker exec aegisai-ollama ollama pull nomic-embed-text
```

### 4. Access the Application

- **Frontend**: http://localhost:5173 (or http://localhost with nginx)
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

### Demo Credentials

| Role | Email | Password |
|------|-------|----------|
| Admin | admin@aegisai.local | admin123 |

## Services

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 5173 | React application |
| Backend API | 8000 | FastAPI REST API |
| PostgreSQL | 5432 | Primary database |
| Qdrant | 6333/6334 | Vector database |
| Ollama | 11434 | LLM inference server |
| Nginx | 80/443 | Reverse proxy (production profile) |

## Development

### Backend Development

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm run test
```

## Project Structure

```
aegisai/
├── backend/
│   ├── app/
│   │   ├── api/v1/           # API routes
│   │   ├── core/             # Config, database, security
│   │   ├── models/           # SQLAlchemy models
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── services/         # Business logic
│   │   └── main.py           # FastAPI application
│   ├── requirements.txt
│   ├── Dockerfile
│   └── alembic/              # Database migrations
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── layouts/          # Page layouts
│   │   ├── pages/            # Page components
│   │   ├── services/         # API services
│   │   ├── stores/           # Zustand stores
│   │   ├── types/            # TypeScript types
│   │   └── App.tsx           # Main app with routing
│   ├── package.json
│   ├── Dockerfile
│   └── tailwind.config.js
├── docker/
│   ├── nginx/                # Nginx configurations
│   └── postgres/             # Database initialization
├── docker-compose.yml
├── .env.example
└── README.md
```

## Configuration

All configuration is via environment variables. See `.env.example` for all options.

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing key (min 32 chars) | **Required** |
| `POSTGRES_PASSWORD` | Database password | **Required** |
| `OLLAMA_MODEL` | Chat model name | `qwen2.5:7b-instruct` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model | `nomic-embed-text` |
| `CHUNK_SIZE` | Document chunk size | `512` |
| `TOP_K_RESULTS` | RAG retrieval count | `5` |

## Document Classification

Documents are classified into four tiers:

| Level | Description | Access |
|-------|-------------|--------|
| `PUBLIC_INTERNAL` | Internal public documents | All authenticated users |
| `CONFIDENTIAL` | Department-sensitive | Department members + managers |
| `RESTRICTED` | Role-sensitive | Specific roles + explicit grants |
| `HIGHLY_RESTRICTED` | Need-to-know basis | Explicit user grants only |

## API Endpoints

### Authentication
- `POST /api/v1/auth/login` - Login
- `POST /api/v1/auth/refresh` - Refresh access token
- `POST /api/v1/auth/logout` - Logout
- `GET /api/v1/auth/me` - Current user

### Documents
- `POST /api/v1/documents` - Upload document
- `GET /api/v1/documents` - List documents (permission-filtered)
- `GET /api/v1/documents/{id}` - Get document details
- `PATCH /api/v1/documents/{id}` - Update document
- `DELETE /api/v1/documents/{id}` - Delete document
- `POST /api/v1/documents/{id}/reindex` - Re-index for RAG
- `GET /api/v1/documents/{id}/permissions` - Get permissions
- `POST /api/v1/documents/{id}/permissions` - Grant permission
- `DELETE /api/v1/documents/{id}/permissions/{perm_id}` - Revoke permission

### Chat
- `POST /api/v1/chat` - Send message (RAG)
- `GET /api/v1/chat/conversations` - List conversations
- `POST /api/v1/chat/conversations` - Create conversation
- `GET /api/v1/chat/conversations/{id}` - Get conversation
- `DELETE /api/v1/chat/conversations/{id}` - Delete conversation
- `GET /api/v1/chat/conversations/{id}/messages` - Get messages

### Users (Admin only)
- `GET /api/v1/users` - List users
- `POST /api/v1/users` - Create user
- `GET /api/v1/users/{id}` - Get user
- `PATCH /api/v1/users/{id}` - Update user
- `DELETE /api/v1/users/{id}` - Delete user
- `GET /api/v1/users/roles/list` - List roles
- `GET /api/v1/users/departments/list` - List departments

### Health
- `GET /api/health` - Basic health
- `GET /api/health/ready` - Readiness check
- `GET /api/health/live` - Liveness check
- `GET /api/health/status` - Detailed system status

## Security

- **Passwords**: Bcrypt with cost factor 12
- **Tokens**: JWT HS256, 30min access / 7day refresh
- **CORS**: Configurable origins
- **Audit Logs**: All security events tracked
- **Rate Limiting**: Recommended via nginx

## License

MIT License - Built for Smart India Hackathon 2024

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

---

**Built with ❤️ for the Smart India Hackathon**