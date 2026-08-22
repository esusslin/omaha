.PHONY: dev up down migrate revision test lint fmt check scan worker chunk embed retrieve-status

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

# --- model work runs in the Linux container (host lacks ML wheels) ---

worker:        ## build the linux image
	docker compose --profile worker build worker

chunk:         ## chunk documents (pure python, runs on host)
	uv run python -m omaha.retrieve.run chunk

embed:         ## embed chunks (container — needs onnxruntime)
	docker compose --profile worker run --rm worker uv run python -m omaha.retrieve.run embed

retrieve-status:
	uv run python -m omaha.retrieve.run status
