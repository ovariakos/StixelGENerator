#!/usr/bin/env python3

"""
Extract images and point clouds from a bag (SQLite .db3 or MCAP) and
pair them by nearest timestamps without adding ROS2 dependencies.
"""

import argparse
import os
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

from mcap.reader import make_reader
import numpy as np

POINT_STEP = 32
OFFSETS = {"x": 0, "y": 4, "z": 8}


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def query_id(cur: sqlite3.Cursor, name: str) -> Optional[int]:
    cur.execute("SELECT id FROM topics WHERE name=?", (name,))
    row = cur.fetchone()
    return row[0] if row else None


def fetch_messages(cur: sqlite3.Cursor, tid: int) -> List[Tuple[int, bytes]]:
    cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (tid,),
    )
    return cur.fetchall()


def fetch_messages_mcap(reader, topic: str) -> List[Tuple[int, bytes]]:
    return [
        (msg.log_time, msg.data)
        for _, _ch, msg in reader.iter_messages(topics=[topic])
    ]


def extract_images(msgs: List[Tuple[int, bytes]], out_dir: str):
    """Extract JPEG images from messages."""
    ensure_dir(out_dir)
    mapping = []
    for idx, (ts, blob) in enumerate(msgs):
        start = blob.find(b"\xff\xd8")
        end = blob.rfind(b"\xff\xd9")
        if start < 0 or end < 0:
            continue
        fname = f"{idx:06d}.jpg"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(blob[start : end + 2])
        mapping.append((idx, ts, os.path.join("images", fname)))
    return mapping


def extract_pointclouds(msgs: List[Tuple[int, bytes]], out_dir: str, step: int = POINT_STEP):
    """Extract XYZ point clouds from messages."""
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


def list_topics_db3(cur: sqlite3.Cursor) -> List[str]:
    cur.execute("SELECT name FROM topics")
    return [row[0] for row in cur.fetchall()]


def list_topics_mcap(reader) -> List[str]:
    summary = reader.get_summary()
    if summary is None:
        return []
    return [ch.topic for ch in summary.channels.values()]


def pair_by_time(
    img_map: List[Tuple[int, int, str]],
    pc_map: List[Tuple[int, int, str]],
    max_dt_ns: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """
    Pair image and pointcloud indices by nearest timestamps.
    Returns list of (img_idx, pc_idx, dt_ns).
    """
    pairs = []
    i = j = 0
    while i < len(img_map) and j < len(pc_map):
        ti = img_map[i][1]
        tj = pc_map[j][1]
        dt = abs(ti - tj)
        if max_dt_ns is None or dt <= max_dt_ns:
            pairs.append((i, j, dt))
            i += 1
            j += 1
        elif ti < tj:
            i += 1
        else:
            j += 1
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert bag to timestamp-paired dataset")
    parser.add_argument("--db", "--bag", dest="bag", required=True,
                        help="rosbag file (.db3 or .mcap)")
    parser.add_argument("--image_topic", required=True, help="Image topic name")
    parser.add_argument("--pc_topic", required=True, help="Pointcloud topic name")
    parser.add_argument("--out", default="rosbag_dataset", help="Output directory")
    parser.add_argument("--max_dt_ms", type=float, default=50.0,
                        help="Maximum allowed Δt between image and pc (ms); use <0 to disable")
    args = parser.parse_args()

    ensure_dir(args.out)
    img_map, pc_map = [], []

    if args.bag.endswith(".db3"):
        conn = sqlite3.connect(args.bag)
        cur = conn.cursor()
        print("Available topics (DB3):")
        for t in list_topics_db3(cur):
            print(" ", t)
        tid_img = query_id(cur, args.image_topic)
        if tid_img is not None:
            msgs = fetch_messages(cur, tid_img)
            img_map = extract_images(msgs, os.path.join(args.out, "images"))
        tid_pc = query_id(cur, args.pc_topic)
        if tid_pc is not None:
            msgs = fetch_messages(cur, tid_pc)
            pc_map = extract_pointclouds(msgs, os.path.join(args.out, "pointclouds"))
        conn.close()

    elif args.bag.endswith(".mcap"):
        print("Available topics (MCAP):")
        with open(args.bag, "rb") as f:
            reader = make_reader(f)
            for t in list_topics_mcap(reader):
                print(" ", t)
            msgs_img = fetch_messages_mcap(reader, args.image_topic)
            img_map = extract_images(msgs_img, os.path.join(args.out, "images"))
            msgs_pc = fetch_messages_mcap(reader, args.pc_topic)
            pc_map = extract_pointclouds(msgs_pc, os.path.join(args.out, "pointclouds"))
    else:
        raise ValueError("Unsupported bag format. Use .db3 or .mcap")

    # Pair by timestamp
    max_dt_ns = None if args.max_dt_ms < 0 else int(args.max_dt_ms * 1e6)
    pairs = pair_by_time(img_map, pc_map, max_dt_ns)

    # Write mapping CSV
    map_file = os.path.join(args.out, "dataset_map.csv")
    with open(map_file, "w") as f:
        f.write("index,dt_ns,image_timestamp,image_file,pc_timestamp,pc_file\n")
        for k, (ii, jj, dt) in enumerate(pairs):
            _, its, ifile = img_map[ii]
            _, pts, pfile = pc_map[jj]
            f.write(f"{k},{dt},{its},{ifile},{pts},{pfile}\n")

    print(f"Paired {len(pairs)} frames (Δt ≤ {args.max_dt_ms} ms).")


if __name__ == "__main__":
    main()
