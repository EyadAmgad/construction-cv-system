"""
End-to-end integration test: cv_service → Kafka → kafka_consumer → PostgreSQL.

Sends N known test payloads to Kafka, runs the consumer to drain only the
NEW messages, then queries PostgreSQL and asserts every payload landed correctly.

Usage:
    python src/services/test_kafka_pipeline.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from kafka import KafkaConsumer, KafkaProducer
from kafka.structs import TopicPartition

ROOT = Path(__file__).parent.parent.parent   # src/services → scripts → project root
sys.path.insert(0, str(ROOT / "src" / "services"))
from kafka_consumer import get_connection, insert_event

KAFKA_BOOTSTRAP = "localhost:9094"
TOPIC           = "equipment-detections"

# Use distinctive frame IDs unlikely to collide with real data
TEST_PAYLOADS = [
    {
        "frame_id": 999001,
        "equipment_id": "TEST-EXCAVATOR",
        "equipment_class": "excavator",
        "timestamp": "00:00:06.666",
        "utilization": {"current_state": "ACTIVE", "current_activity": "DIGGING"},
        "time_analytics": {
            "total_tracked_seconds": 6.667,
            "total_active_seconds": 6.667,
            "total_idle_seconds": 0.0,
            "utilization_percent": 100.0,
        },
    },
    {
        "frame_id": 999002,
        "equipment_id": "TEST-EQ-1",
        "equipment_class": "truck",
        "timestamp": "00:00:13.333",
        "utilization": {"current_state": "ACTIVE", "current_activity": "LOADING"},
        "time_analytics": {
            "total_tracked_seconds": 13.333,
            "total_active_seconds": 10.0,
            "total_idle_seconds": 3.333,
            "utilization_percent": 75.0,
        },
    },
    {
        "frame_id": 999003,
        "equipment_id": "TEST-EXCAVATOR",
        "equipment_class": "excavator",
        "timestamp": "00:00:30.000",
        "utilization": {"current_state": "INACTIVE", "current_activity": "WAITING"},
        "time_analytics": {
            "total_tracked_seconds": 15.0,
            "total_active_seconds": 12.5,
            "total_idle_seconds": 2.5,
            "utilization_percent": 83.3,
        },
    },
]

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"


def get_end_offsets(topic: str) -> dict:
    """Return the current end offsets for all partitions of a topic."""
    tmp = KafkaConsumer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=None,
        consumer_timeout_ms=3000,
    )
    partitions = tmp.partitions_for_topic(topic) or {0}
    tps = [TopicPartition(topic, p) for p in partitions]
    tmp.assign(tps)
    offsets = tmp.end_offsets(tps)
    tmp.close()
    return offsets


def publish_test_messages(end_offsets: dict) -> int:
    print(f"[1/3] Publishing {len(TEST_PAYLOADS)} test messages to Kafka topic '{TOPIC}'...")
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
    )
    for p in TEST_PAYLOADS:
        producer.send(TOPIC, value=p)
    producer.flush()
    producer.close()
    print(f"      Published {len(TEST_PAYLOADS)} messages.")
    return len(TEST_PAYLOADS)


def consume_only_new(start_offsets: dict) -> int:
    """Consume only messages published after start_offsets, insert into DB."""
    print("[2/3] Consuming only new messages and inserting into PostgreSQL...")
    consumer = KafkaConsumer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=None,                          # no group — manual offset control
        enable_auto_commit=False,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        consumer_timeout_ms=6000,
    )
    consumer.assign(list(start_offsets.keys()))
    # Seek each partition to the offset recorded before publishing
    for tp, offset in start_offsets.items():
        consumer.seek(tp, offset)

    conn = get_connection()
    inserted = 0
    with conn.cursor() as cur:
        for msg in consumer:
            insert_event(cur, msg.value)
            inserted += 1
        conn.commit()
    conn.close()
    consumer.close()
    print(f"      Inserted {inserted} event(s).")
    return inserted


def verify_db() -> int:
    print("[3/3] Verifying PostgreSQL data for test payloads...")
    conn = get_connection()
    failures = 0

    with conn.cursor() as cur:
        for expected in TEST_PAYLOADS:
            cur.execute(
                """
                SELECT frame_id, equipment_id, equipment_class, video_timestamp,
                       current_state, current_activity,
                       total_tracked_seconds, total_active_seconds,
                       total_idle_seconds, utilization_percent
                FROM equipment_events
                WHERE frame_id = %s AND equipment_id = %s
                ORDER BY received_at DESC
                LIMIT 1
                """,
                (expected["frame_id"], expected["equipment_id"]),
            )
            row = cur.fetchone()
            if row is None:
                print(f"  {FAIL}  frame_id={expected['frame_id']} "
                      f"equipment_id={expected['equipment_id']} — NOT FOUND in DB")
                failures += 1
                continue

            (frame_id, equipment_id, equipment_class, video_timestamp,
             current_state, current_activity,
             total_tracked_seconds, total_active_seconds,
             total_idle_seconds, utilization_percent) = row

            checks = [
                ("frame_id",              frame_id,              expected["frame_id"]),
                ("equipment_id",          equipment_id,          expected["equipment_id"]),
                ("equipment_class",       equipment_class,       expected["equipment_class"]),
                ("video_timestamp",       video_timestamp,       expected["timestamp"]),
                ("current_state",         current_state,         expected["utilization"]["current_state"]),
                ("current_activity",      current_activity,      expected["utilization"]["current_activity"]),
                ("total_tracked_seconds", round(total_tracked_seconds, 3),
                                                                  expected["time_analytics"]["total_tracked_seconds"]),
                ("total_active_seconds",  round(total_active_seconds, 3),
                                                                  expected["time_analytics"]["total_active_seconds"]),
                ("total_idle_seconds",    round(total_idle_seconds, 3),
                                                                  expected["time_analytics"]["total_idle_seconds"]),
                ("utilization_percent",   round(utilization_percent, 1),
                                                                  expected["time_analytics"]["utilization_percent"]),
            ]

            row_ok = True
            for field, got, want in checks:
                match = abs(float(got) - float(want)) < 0.01 if isinstance(want, float) else got == want
                if not match:
                    print(f"  {FAIL}  frame={expected['frame_id']} "
                          f"{field}: expected={want!r}  got={got!r}")
                    failures += 1
                    row_ok = False

            if row_ok:
                print(f"  {PASS}  frame_id={frame_id}  {equipment_id:16s}  "
                      f"{current_activity:10s}  util={utilization_percent:.1f}%")

    conn.close()
    return failures


def cleanup_test_rows():
    """Remove test rows so the test is repeatable."""
    conn = get_connection()
    test_ids = [p["equipment_id"] for p in TEST_PAYLOADS]
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM equipment_events WHERE equipment_id = ANY(%s)", (test_ids,)
        )
        deleted = cur.rowcount
    conn.commit()
    conn.close()
    print(f"      Cleaned up {deleted} test row(s) from DB.")


def main():
    print("=" * 60)
    print("  Kafka → PostgreSQL Integration Test")
    print("=" * 60)

    # Record current offsets BEFORE publishing so we only consume our messages
    end_offsets = get_end_offsets(TOPIC)
    publish_test_messages(end_offsets)
    time.sleep(1)
    inserted = consume_only_new(end_offsets)
    failures = verify_db()
    print()
    cleanup_test_rows()

    print()
    print("=" * 60)
    if failures == 0 and inserted >= len(TEST_PAYLOADS):
        print(f"\033[92m  ALL {len(TEST_PAYLOADS)} TESTS PASSED\033[0m")
    else:
        print(f"\033[91m  {failures} check(s) FAILED  (inserted={inserted})\033[0m")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
