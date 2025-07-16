#!/usr/bin/env python3
"""Extract images and point clouds from a ROS2 bag.

This utility reads a rosbag2 SQLite database or an MCAP file and writes each
image and point cloud message into a separate file. The output can be used as
the raw dataset for StixelGENerator after adding calibration information.
"""
import argparse
import os
import sqlite3
from pathlib import Path
from typing import List

from mcap.reader import make_reader
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


def fetch_messages_mcap(reader, topic: str):
    return [
        (msg.log_time, msg.data)
        for _, _ch, msg in reader.iter_messages(topics=[topic])
    ]


def extract_images(msgs, out_dir: str):
    """Extract JPEG images from messages.

    Args:
        msgs: Sequence of (timestamp, data) tuples.
        out_dir: Destination directory for image files.

    Returns:
        List of tuples ``(index, timestamp, filename)`` for mapping creation.
    """
    ensure_dir(out_dir)
    mapping = []
    for idx, (ts, blob) in enumerate(msgs):
        start = blob.find(b"\xff\xd8")
        end = blob.rfind(b"\xff\xd9")
        if start < 0 or end < 0:
            continue
        fname = f"{idx:06d}.jpg"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(blob[start:end + 2])
        mapping.append((idx, ts, os.path.join("images", fname)))
    return mapping


def extract_pointclouds(msgs, out_dir: str, step: int = POINT_STEP):
    """Extract XYZ point clouds from messages.

    Args:
        msgs: Sequence of (timestamp, data) tuples.
        out_dir: Destination directory for CSV files.
        step: Point step size in bytes.

    Returns:
        List of tuples ``(index, timestamp, filename)`` for mapping creation.
    """
    ensure_dir(out_dir)
    mapping = []
    for idx, (ts, blob) in enumerate(msgs):
        hdr = next((i for i in range(1024) if (len(blob) - i) % step == 0), None)
        if hdr is None:
            continue
        data = blob[hdr:]
        pts = np.frombuffer(data, dtype=np.float32)
        arr = pts.reshape(len(data) // step, step // 4)
        xyz = arr[:, [OFFSETS["x"] // 4, OFFSETS["y"] // 4, OFFSETS["z"] // 4]]
        fname = f"{idx:06d}.csv"
        np.savetxt(
            os.path.join(out_dir, fname),
            xyz,
            delimiter=",",
            header="x,y,z",
            comments="",
        )
        mapping.append((idx, ts, os.path.join("pointclouds", fname)))
    return mapping


def list_topics_db3(cur: sqlite3.Cursor):
    cur.execute("SELECT name FROM topics")
    return [row[0] for row in cur.fetchall()]


def list_topics_mcap(reader) -> list[str]:
    summary = reader.get_summary()
    if summary is None:
        return []
    return [ch.topic for ch in summary.channels.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert rosbag to dataset")
    parser.add_argument(
        "--db",
        "--bag",
        dest="bag",
        required=True,
        help="rosbag file (.db3 or .mcap)",
    )
    parser.add_argument("--image_topic", required=True, help="Image topic name")
    parser.add_argument("--pc_topic", required=True, help="Pointcloud topic name")
    parser.add_argument("--out", default="rosbag_dataset", help="Output directory")
    args = parser.parse_args()

    ensure_dir(args.out)

    img_map: list[tuple[int, int, str]] = []
    pc_map: list[tuple[int, int, str]] = []

    if args.bag.endswith(".db3"):
        conn = sqlite3.connect(args.bag)
        cur = conn.cursor()
        topics = list_topics_db3(cur)
        print("Available topics:")
        for t in topics:
            print(" ", t)

        tid = query_id(cur, args.image_topic)
        if tid is not None:
            msgs = fetch_messages(cur, tid)
            img_map = extract_images(msgs, os.path.join(args.out, "images"))

        tid = query_id(cur, args.pc_topic)
        if tid is not None:
            msgs = fetch_messages(cur, tid)
            pc_map = extract_pointclouds(
                msgs, os.path.join(args.out, "pointclouds")
            )
        conn.close()

    elif args.bag.endswith(".mcap"):
        with open(args.bag, "rb") as f:
            reader = make_reader(f)
            topics = list_topics_mcap(reader)
            print("Available topics:")
            for t in topics:
                print(" ", t)

            if args.image_topic in topics:
                msgs = fetch_messages_mcap(reader, args.image_topic)
                img_map = extract_images(
                    msgs, os.path.join(args.out, "images")
                )
            if args.pc_topic in topics:
                msgs = fetch_messages_mcap(reader, args.pc_topic)
                pc_map = extract_pointclouds(
                    msgs, os.path.join(args.out, "pointclouds")
                )
    else:
        raise ValueError("Unsupported bag format. Use .db3 or .mcap")

    map_file = os.path.join(args.out, "dataset_map.csv")
    with open(map_file, "w") as f:
        f.write("index,image_timestamp,image_file,pc_timestamp,pc_file\n")
        length = min(len(img_map), len(pc_map))
        if len(img_map) != len(pc_map):
            print(
                f"Warning: unmatched message counts - keeping {length} paired frames"
            )
        for idx in range(length):
            _, img_ts, img_file = img_map[idx]
            _, pc_ts, pc_file = pc_map[idx]
            f.write(f"{idx},{img_ts},{img_file},{pc_ts},{pc_file}\n")


if __name__ == "__main__":
    main()
