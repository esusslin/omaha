.PHONY: dev up down migrate revision test lint fmt check scan worker chunk embed retrieve-status

up:            ## start postgres
	docker compose up -d

down:
	docker compose down

migrate:       ## apply migrations
	uv run alembic upgrade head

revision:      ## make a migration: make revision m="add foo"
	uv run alembic revision --autogenerate -m "$(m)"

dev:           ## run the api on the host — search works, but lexical only
	uv run uvicorn omaha.api:app --reload

# The worker declares no ports, so publish one explicitly rather than --service-ports.
# SCHEDULER_ENABLED is already false for this service in compose: a demo process should
# not also be fetching from 32 club sites.
demo:          ## run the api in the container so /ui has dense retrieval too
	docker compose --profile worker run --rm -p 8000:8000 worker \
		uv run uvicorn omaha.api:app --host 0.0.0.0 --port 8000

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

audit:         ## extraction quality, counted
	uv run python -m omaha.retrieve.run audit

search:        ## hybrid search: make search q="who is out with a foot injury"
	docker compose --profile worker run --rm worker \
		uv run python -m omaha.retrieve.search_cli --query "$(q)"

eval:          ## compare lexical / dense / hybrid over the gold set (container)
	docker compose --profile worker run --rm worker \
		uv run python -m omaha.eval.run --mode all --show-misses

eval-lexical:  ## lexical only — no model needed, runs on the host
	uv run python -m omaha.eval.run --mode lexical --show-misses
