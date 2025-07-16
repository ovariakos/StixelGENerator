#!/usr/bin/env python3
"""Extract images and point clouds from a ROS2 bag.

This utility reads a rosbag2 SQLite database and writes each image and
point cloud message into a separate file. The output can be used as the
raw dataset for StixelGENerator after adding calibration information.
"""
import argparse
import os
import sqlite3
from pathlib import Path
import numpy as np

POINT_STEP = 32
OFFSETS = {"x": 0, "y": 4, "z": 8}


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def query_id(cur: sqlite3.Cursor, name: str):
    cur.execute("SELECT id FROM topics WHERE name=?", (name,))
    row = cur.fetchone()
    return row[0] if row else None


def fetch_messages(cur: sqlite3.Cursor, tid: int):
    cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (tid,),
    )
    return cur.fetchall()


def extract_images(msgs, out_dir: str) -> None:
    ensure_dir(out_dir)
    for ts, blob in msgs:
        start = blob.find(b"\xff\xd8")
        end = blob.rfind(b"\xff\xd9")
        if start < 0 or end < 0:
            continue
        with open(os.path.join(out_dir, f"{ts}.jpg"), "wb") as f:
            f.write(blob[start:end + 2])


def extract_pointclouds(msgs, out_dir: str, step: int = POINT_STEP) -> None:
    ensure_dir(out_dir)
    for ts, blob in msgs:
        hdr = next((i for i in range(1024) if (len(blob) - i) % step == 0), None)
        if hdr is None:
            continue
        data = blob[hdr:]
        pts = np.frombuffer(data, dtype=np.float32)
        arr = pts.reshape(len(data) // step, step // 4)
        xyz = arr[:, [OFFSETS["x"] // 4, OFFSETS["y"] // 4, OFFSETS["z"] // 4]]
        np.savetxt(
            os.path.join(out_dir, f"{ts}.csv"),
            xyz,
            delimiter=",",
            header="x,y,z",
            comments="",
        )


def list_topics(cur: sqlite3.Cursor):
    cur.execute("SELECT name FROM topics")
    return [row[0] for row in cur.fetchall()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert rosbag2 DB to dataset")
    parser.add_argument("--db", required=True, help="sqlite3 bag file (*.db3)")
    parser.add_argument("--image_topic", required=True, help="Image topic name")
    parser.add_argument("--pc_topic", required=True, help="Pointcloud topic name")
    parser.add_argument("--out", default="rosbag_dataset", help="Output directory")
    args = parser.parse_args()

    ensure_dir(args.out)

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    print("Available topics:")
    for t in list_topics(cur):
        print(" ", t)

    tid = query_id(cur, args.image_topic)
    if tid is not None:
        msgs = fetch_messages(cur, tid)
        extract_images(msgs, os.path.join(args.out, "images"))

    tid = query_id(cur, args.pc_topic)
    if tid is not None:
        msgs = fetch_messages(cur, tid)
        extract_pointclouds(msgs, os.path.join(args.out, "pointclouds"))

    conn.close()


if __name__ == "__main__":
    main()
