#!/bin/bash
echo 'Starting Construction CV System...'

# 1. Start Docker containers (PostgreSQL, Kafka, pgAdmin)
echo 'Starting databases and message queues...'
docker compose up -d

# 2. Start the Kafka Consumer in the background to listen for CV events
echo 'Starting Kafka Consumer...'
/home/eyada/miniconda3/envs/edu-mentor/bin/python src/services/kafka_consumer.py &

# 3. Start the FastAPI Backend (which also serves the Frontend UI)
echo 'Starting FastAPI UI Server...'
/home/eyada/miniconda3/envs/edu-mentor/bin/python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000

# When uvicorn is stopped, make sure to clean up the consumer
wait
