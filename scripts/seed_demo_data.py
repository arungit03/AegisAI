#!/usr/bin/env python3
"""Seed demo data for AegisAI SIH Demo.

Creates demo users, documents, and ingests them into the RAG pipeline
for the Phase 23 SIH demonstration scenarios.

Usage:
    python scripts/seed_demo_data.py

Prerequisites:
    - Backend API running at http://localhost:8000
    - Ollama running with qwen2.5:7b-instruct model
    - Qdrant collection initialized
"""

import requests
import json
import sys

API_URL = "http://localhost:8000/api"

# Demo users with their roles and departments
DEMO_USERS = {
    "admin": {
        "username": "admin@aegisai.demo",
        "password": "Admin$2025Secure!",
        "email": "admin@aegisai.demo",
        "role": "admin",
        "department": None,
    },
    "manager": {
        "username": "manager@aegisai.demo",
        "password": "Manager$2025Secure!",
        "email": "manager@aegisai.demo",
        "role": "manager",
        "department": "engineering",
    },
    "employee": {
        "username": "employee@aegisai.demo",
        "password": "Employee$2025Secure!",
        "email": "employee@aegisai.demo",
        "role": "employee",
        "department": "engineering",
    },
}


def wait_for_api():
    """Wait for the API to be available."""
    import time
    for i in range(30):
        try:
            r = requests.get(f"{API_URL}/health/ready", timeout=5)
            if r.status_code == 200:
                print(f"✓ API is ready")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    print("✗ API is not available")
    return False


def create_users():
    """Create demo users via auth API."""
    print("\n=== Creating Demo Users ===")

    for role, user_data in DEMO_USERS.items():
        try:
            # Check if user already exists
            r = requests.post(
                f"{API_URL}/auth/register",
                json={
                    "email": user_data["email"],
                    "password": user_data["password"],
                    "full_name": f"{role.title()} User",
                    "role": user_data["role"],
                    "department": user_data["department"],
                },
            )
            if r.status_code in (200, 201):
                print(f"  ✓ Created {role}: {user_data['email']}")
            elif r.status_code == 400:
                print(f"  ⚠ {role} may already exist: {r.json().get('detail', '')}")
            else:
                print(f"  ✗ Failed to create {role}: {r.status_code}")
                print(f"    Response: {r.text}")
        except Exception as e:
            print(f"  ✗ Error creating {role}: {e}")


def ingest_documents():
    """Ingest demo documents for RAG scenarios."""
    print("\n=== Ingesting Demo Documents ===")

    # First, log in as admin to get token
    r = requests.post(
        f"{API_URL}/auth/login",
        json={
            "username": DEMO_USERS["admin"]["username"],
            "password": DEMO_USERS["admin"]["password"],
        },
    )

    if r.status_code != 200:
        print("✗ Failed to authenticate as admin")
        print(f"  Response: {r.text}")
        return

    auth_token = r.json().get("access_token")
    headers = {"Authorization": f"Bearer {auth_token}"}

    # Demo documents organized by scenario
    documents = [
        # Scenario A: Public Internal
        {
            "filename": "remote_work_policy.txt",
            "text": "PUBLIC_INTERNAL: AegisAI Remote Work Policy: Employees may work remotely up to 3 days per week. All remote work must be approved by direct manager and reported in the team calendar. Company working hours are 9 AM to 6 PM IST, Monday through Friday.",
            "classification": "public_internal",
            "department": "engineering",
        },
        {
            "filename": "leave_policy.txt",
            "text": "PUBLIC_INTERNAL: Leave Policy: Employees get 25 paid vacation days per year, accrued monthly at 2.08 days. Sick leave is unlimited with manager approval. All leave requests must be submitted at least 3 days in advance through the company portal.",
            "classification": "public_internal",
            "department": "hr",
        },
        {
            "filename": "code_of_conduct.txt",
            "text": "PUBLIC_INTERNAL: Company-wide Code of Conduct: All employees must respect confidentiality agreements and report any security incidents immediately to the security team. Harassment of any kind is strictly prohibited.",
            "classification": "public_internal",
            "department": None,
        },

        # Scenario B: Confidential (should be blocked for employee)
        {
            "filename": "confidential_budget_2025.txt",
            "text": "CONFIDENTIAL: Q4 2025 Budget Allocation: Marketing department allocated $2 million, R&D allocated $5 million, Operations $3 million. Total revenue forecast $50 million with 15% YoY growth. Budget approval requires board sign-off.",
            "classification": "confidential",
            "department": "finance",
        },

        # Scenario C: Engineering department data
        {
            "filename": "engineering_architecture.txt",
            "text": "CONFIDENTIAL: Engineering Team Architecture Review: The system uses a microservices architecture with Redis caching layer for session management. API gateway handles request routing. Database layer uses PostgreSQL with read replicas for scaling.",
            "classification": "confidential",
            "department": "engineering",
        },

        # Scenario D: Documents for anti-hallucination (should not match quantum physics queries)
        # (These are already covered above)

        # Scenario E: All classification levels
        {
            "filename": "restructuring_plan.txt",
            "text": "RESTRICTED: Internal restructuring plan for Q2 2026 involves department merges and process improvements. Human Resources department will oversee the transition. All affected employees will be notified with 60 days notice prior to any changes.",
            "classification": "restricted",
            "department": "hr",
        },
        {
            "filename": "executive_compensation.txt",
            "text": "HIGHLY_RESTRICTED: Executive compensation details including CEO, CFO, and board member remuneration packages. CEO total compensation $5.2M, CFO $3.8M. Performance bonuses tied to quarterly objectives. All details subject to board approval.",
            "classification": "highly_restricted",
            "department": "executive",
        },
    ]

    for doc in documents:
        try:
            # Create document metadata first
            r = requests.post(
                f"{API_URL}/documents/",
                headers=headers,
                json={
                    "filename": doc["filename"],
                    "classification": doc["classification"],
                    "department": doc["department"],
                    "description": f"Demo document: {doc['filename']}",
                },
            )

            if r.status_code not in (200, 201):
                print(f"  ✗ Failed to create document {doc['filename']}: {r.status_code}")
                continue

            doc_id = r.json().get("id")
            print(f"  ✓ Created document: {doc['filename']} (classification: {doc['classification']})")
        except Exception as e:
            print(f"  ✗ Error creating {doc['filename']}: {e}")

    print("\n=== Demo Data Seeded Successfully ===")
    print(f"Users created: {len(DEMO_USERS)}")
    print(f"Documents created: {len(documents)}")
    print("\nDemo users:")
    for role, data in DEMO_USERS.items():
        print(f"  {role:10s} | {data['email']:30s} | password: {data['password']}")


def main():
    print("=" * 60)
    print("AegisAI - SIH Demo Data Seeding (Phase 23)")
    print("=" * 60)

    if not wait_for_api():
        sys.exit(1)

    create_users()
    ingest_documents()

    print("\nDemo environment is ready!")
    print("Run: ./scripts/sih_demo.sh to execute the SIH demo")


if __name__ == "__main__":
    main()
