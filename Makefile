.PHONY: dev up down migrate revision test lint fmt check scan

up:            ## start postgres
	docker compose up -d

down:
	docker compose down

migrate:       ## apply migrations
	uv run alembic upgrade head

revision:      ## make a migration: make revision m="add foo"
	uv run alembic revision --autogenerate -m "$(m)"

dev:           ## run the api
	uv run uvicorn omaha.api:app --reload

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run mypy src

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

check: lint test

scan:          ## secret scan the whole history before going public
	gitleaks detect --source . --verbose
