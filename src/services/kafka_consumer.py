"""
Kafka Consumer — construction CV system.

Listens on the 'equipment-detections' topic and persists each event
into the PostgreSQL equipment_events table.

Pipeline:  cv_service  →  Kafka  →  kafka_consumer  →  PostgreSQL

Usage:
    python src/services/kafka_consumer.py
    python src/services/kafka_consumer.py --topic equipment-detections --bootstrap localhost:9094
"""

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg2
from kafka import KafkaConsumer

ROOT = Path(__file__).parent.parent.parent   # src/services → scripts → project root

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9094")
KAFKA_TOPIC     = os.getenv("KAFKA_TOPIC", "equipment-detections")
KAFKA_GROUP_ID  = "construction-cv-consumer"


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "construction_cv"),
        user=os.getenv("DB_USER", "cvuser"),
        password=os.getenv("DB_PASSWORD", "cvpass"),
    )


def insert_event(cur, msg: dict) -> None:
    cur.execute(
        """
        INSERT INTO equipment_events (
            frame_id, equipment_id, equipment_class, video_timestamp,
            current_state, current_activity,
            total_tracked_seconds, total_active_seconds,
            total_idle_seconds, utilization_percent
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            msg["frame_id"],
            msg["equipment_id"],
            msg["equipment_class"],
            msg["timestamp"],
            msg["utilization"]["current_state"],
            msg["utilization"]["current_activity"],
            msg["time_analytics"]["total_tracked_seconds"],
            msg["time_analytics"]["total_active_seconds"],
            msg["time_analytics"]["total_idle_seconds"],
            msg["time_analytics"]["utilization_percent"],
        ),
    )


def run(bootstrap: str, topic: str) -> None:
    print(f"Connecting to Kafka at {bootstrap}, topic={topic}")
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    print("Kafka consumer ready. Waiting for messages...")

    conn = get_connection()
    conn.autocommit = False
    batch: list[dict] = []

    try:
        for kafka_msg in consumer:
            msg = kafka_msg.value
            batch.append(msg)

            # Commit to DB every 50 messages (or immediately for small batches)
            if len(batch) >= 50:
                with conn.cursor() as cur:
                    for m in batch:
                        insert_event(cur, m)
                conn.commit()
                print(f"✅ Committed {len(batch)} events to PostgreSQL")
                batch.clear()

    except KeyboardInterrupt:
        print("\nStopping consumer...")
    finally:
        # Flush remaining
        if batch:
            with conn.cursor() as cur:
                for m in batch:
                    insert_event(cur, m)
            conn.commit()
            print(f"✅ Flushed {len(batch)} remaining events to PostgreSQL")
        conn.close()
        consumer.close()
        print("Consumer stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP,
                        help="Kafka bootstrap servers (default: localhost:9094)")
    parser.add_argument("--topic", default=KAFKA_TOPIC,
                        help="Kafka topic to consume (default: equipment-detections)")
    args = parser.parse_args()
    run(args.bootstrap, args.topic)
