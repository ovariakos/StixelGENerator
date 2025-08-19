#!/usr/bin/env python3

"""
Extract images and point clouds from a bag (SQLite .db3 or MCAP) and
pair them by nearest timestamps without adding ROS2 dependencies.

(Final version with robust PointCloud2 parsing for Ouster LiDAR data)
"""

import argparse
import os
import sqlite3
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from mcap.reader import make_reader


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def query_id(cur: sqlite3.Cursor, name: str) -> Optional[int]:
    """Query the topic ID for a given topic name in a .db3 file."""
    cur.execute("SELECT id FROM topics WHERE name=?", (name,))
    row = cur.fetchone()
    return row[0] if row else None


def fetch_messages(cur: sqlite3.Cursor, tid: int) -> List[Tuple[int, bytes]]:
    """Fetch all messages for a given topic ID from a .db3 file."""
    cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (tid,),
    )
    return cur.fetchall()


def fetch_messages_mcap(reader, topic: str) -> List[Tuple[int, bytes]]:
    """Fetch all messages for a given topic from an MCAP reader."""
    return [
        (msg.log_time, msg.data)
        for _, _ch, msg in reader.iter_messages(topics=[topic])
    ]


def extract_images(msgs: List[Tuple[int, bytes]], out_dir: str):
    """Extract JPEG images from a list of messages."""
    ensure_dir(out_dir)
    mapping = []
    for idx, (ts, blob) in enumerate(msgs):
        start = blob.find(b"\xff\xd8")
        end = blob.rfind(b"\xff\xd9")
        if start < 0 or end < 0:
            continue
        fname = f"{idx:06d}.jpg"
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(blob[start: end + 2])
        mapping.append((idx, ts, os.path.join("images", fname)))
    return mapping


# --- FINAL VERSION: ROBUST MANUAL CDR PARSER ---
def extract_pointclouds(msgs: List[Tuple[int, bytes]], out_dir: str):
    """
    Extracts XYZ point clouds via manual parsing of the ROS2 CDR format.
    This version is tailored to the specific non-standard bag file provided.
    """
    ensure_dir(out_dir)
    mapping = []

    for idx, (ts, blob) in enumerate(msgs):
        try:
            # The data has a 4-byte prefix (00 01 00 00) which must be skipped.
            if not blob.startswith(b'\x00\x01\x00\x00'):
                continue  # Skip if the prefix is missing
            msg_blob = blob[4:]

            cursor = 0

            # --- 1. Read Header ---
            # stamp (2 * uint32), frame_id (string)
            cursor += 8  # Skip timestamp
            frame_id_len = struct.unpack_from('<I', msg_blob, cursor)[0]
            cursor += 4 + frame_id_len
            # Align cursor to the next 4-byte boundary after the string
            cursor += (4 - (cursor % 4)) % 4

            # --- 2. Read PointCloud2 Metadata ---
            height, width = struct.unpack_from('<II', msg_blob, cursor)
            cursor += 8

            fields_len = struct.unpack_from('<I', msg_blob, cursor)[0]
            cursor += 4

            field_defs = {}
            for _ in range(fields_len):
                name_len = struct.unpack_from('<I', msg_blob, cursor)[0]
                cursor += 4
                name = msg_blob[cursor: cursor + name_len].partition(b'\0')[0].decode('ascii')
                cursor += name_len
                # Align after string
                cursor += (4 - (cursor % 4)) % 4

                offset = struct.unpack_from('<I', msg_blob, cursor)[0]
                cursor += 4
                datatype = struct.unpack_from('<B', msg_blob, cursor)[0]
                cursor += 1
                # Align after uint8
                cursor += (4 - (cursor % 4)) % 4
                count = struct.unpack_from('<I', msg_blob, cursor)[0]
                cursor += 4

                if datatype == 7:  # We only care about FLOAT32 fields
                    field_defs[name] = offset

            is_bigendian = struct.unpack_from('<B', msg_blob, cursor)[0]
            cursor += 1
            # Align after bool/uint8
            cursor += (4 - (cursor % 4)) % 4
            point_step, row_step = struct.unpack_from('<II', msg_blob, cursor)
            cursor += 8

            # --- 3. Read Point Data ---
            data_len = struct.unpack_from('<I', msg_blob, cursor)[0]
            cursor += 4
            point_data = msg_blob[cursor: cursor + data_len]

            if 'x' not in field_defs or 'y' not in field_defs or 'z' not in field_defs:
                continue

            # --- 4. Unpack and Save Points ---
            points = []
            endian_char = '>' if is_bigendian else '<'
            x_off, y_off, z_off = field_defs['x'], field_defs['y'], field_defs['z']

            num_points = data_len // point_step
            for i in range(num_points):
                base_offset = i * point_step
                try:
                    px = struct.unpack_from(f'{endian_char}f', point_data, base_offset + x_off)[0]
                    py = struct.unpack_from(f'{endian_char}f', point_data, base_offset + y_off)[0]
                    pz = struct.unpack_from(f'{endian_char}f', point_data, base_offset + z_off)[0]

                    if px != 0.0 or py != 0.0 or pz != 0.0:
                        points.append([px, py, pz])
                except struct.error:
                    continue

            if not points:
                continue

            xyz_array = np.array(points, dtype=np.float32)
            fname = f"{idx:06d}.csv"
            save_path = os.path.join(out_dir, fname)
            np.savetxt(save_path, xyz_array, delimiter=",", header="x,y,z", comments="")
            mapping.append((idx, ts, os.path.join("pointclouds", fname)))

        except Exception as e:
            print(f"Warning: Could not process pointcloud message {idx}. Error: {e}")
            continue

    return mapping


def list_topics_db3(cur: sqlite3.Cursor) -> List[str]:
    """List all topics in a .db3 file."""
    cur.execute("SELECT name FROM topics")
    return [row[0] for row in cur.fetchall()]


def list_topics_mcap(reader) -> List[str]:
    """List all topics in an MCAP file."""
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
    if not img_map or not pc_map:
        return []

    pairs = []
    img_timestamps = np.array([item[1] for item in img_map])
    pc_timestamps = np.array([item[1] for item in pc_map])

    for i, img_item in enumerate(img_map):
        img_ts = img_item[1]
        insert_idx = np.searchsorted(pc_timestamps, img_ts, side='left')

        if insert_idx == 0:
            best_pc_idx_in_map = 0
        elif insert_idx == len(pc_timestamps):
            best_pc_idx_in_map = len(pc_timestamps) - 1
        else:
            dt_before = abs(img_ts - pc_timestamps[insert_idx - 1])
            dt_after = abs(img_ts - pc_timestamps[insert_idx])
            if dt_before < dt_after:
                best_pc_idx_in_map = insert_idx - 1
            else:
                best_pc_idx_in_map = insert_idx

        pc_item = pc_map[best_pc_idx_in_map]
        dt = abs(img_ts - pc_item[1])

        if max_dt_ns is None or dt <= max_dt_ns:
            pairs.append((img_item[0], pc_item[0], dt))

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert bag to timestamp-paired dataset")
    parser.add_argument("--db", "--bag", dest="bag", required=True, help="rosbag file (.db3 or .mcap)")
    parser.add_argument("--image_topic", required=True, help="Image topic name")
    parser.add_argument("--pc_topic", required=True, help="Pointcloud topic name")
    parser.add_argument("-o", "--out", default="rosbag_dataset", help="Output directory")
    parser.add_argument("--max_dt_ms", type=float, default=100.0,
                        help="Maximum allowed Δt between image and pc (ms); use <0 to disable")
    args = parser.parse_args()

    ensure_dir(args.out)
    img_map, pc_map = [], []

    if args.bag.endswith(".db3"):
        conn = sqlite3.connect(args.bag)
        cur = conn.cursor()
        print("Available topics (DB3):")
        for t in list_topics_db3(cur):
            print("  ", t)
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
        with open(args.bag, "rb") as f:
            reader = make_reader(f)
            print("Available topics (MCAP):")
            for t in list_topics_mcap(reader):
                print("  ", t)
            msgs_img = fetch_messages_mcap(reader, args.image_topic)
            img_map = extract_images(msgs_img, os.path.join(args.out, "images"))
            msgs_pc = fetch_messages_mcap(reader, args.pc_topic)
            pc_map = extract_pointclouds(msgs_pc, os.path.join(args.out, "pointclouds"))
    else:
        raise ValueError("Unsupported bag format. Use .db3 or .mcap")

    if not img_map:
        print(f"Warning: No images found on topic '{args.image_topic}'")
    if not pc_map:
        print(f"Warning: No pointclouds extracted from topic '{args.pc_topic}'")

    max_dt_ns = None if args.max_dt_ms < 0 else int(args.max_dt_ms * 1e6)
    img_map_dict = {item[0]: item for item in img_map}
    pc_map_dict = {item[0]: item for item in pc_map}

    pairs = pair_by_time(img_map, pc_map, max_dt_ns)

    map_file = os.path.join(args.out, "dataset_map.csv")
    with open(map_file, "w") as f:
        f.write("index,dt_ns,image_timestamp,image_file,pc_timestamp,pc_file\n")
        for k, (ii, jj, dt) in enumerate(pairs):
            if ii in img_map_dict and jj in pc_map_dict:
                _, its, ifile = img_map_dict[ii]
                _, pts, pfile = pc_map_dict[jj]
                f.write(f"{k},{dt},{its},{ifile.replace(os.sep, '/')},{pts},{pfile.replace(os.sep, '/')}\n")

    print(f"\nExtracted {len(img_map)} images and {len(pc_map)} pointclouds.")
    print(f"Paired {len(pairs)} frames (Δt ≤ {args.max_dt_ms} ms).")
    print(f"Dataset written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()