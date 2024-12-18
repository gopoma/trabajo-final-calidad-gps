run-dev:
	docker compose build && docker-compose up
run-prod:
	docker compose build && docker-compose -f docker-compose.prod.yaml --env-file .env.prod up --build
