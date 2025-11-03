import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
import vrplib
import os


@dataclass
class VRPInstance:
    name: str
    capacity: int
    dimension: int
    coordinates: List[Tuple[float, float]]  # index 1..n (0 reserved for depot)
    demands: List[int]  # index 1..n (0 reserved for depot)
    depot_index: int
    distance_matrix: List[List[float]]


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_distance_matrix(coords: List[Tuple[float, float]]) -> List[List[float]]:
    n = len(coords)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dij = _euclidean(coords[i], coords[j])
            d[i][j] = dij
            d[j][i] = dij
    return d


def load_tsplib_vrp(path: str) -> VRPInstance:
    data = vrplib.read_instance(path)
    # vrplib fields are typically 1-indexed
    name = data.get("name", "")
    capacity = int(data["capacity"]) if "capacity" in data else 0

    # Depot can be list/array
    depot_field = data.get("depot", [1])
    if isinstance(depot_field, (list, tuple)):
        depot_original = int(depot_field[0]) if depot_field else 1
    else:
        try:
            depot_original = int(depot_field)
        except Exception:
            depot_original = 1

    # Node coordinates may be dict (id -> (x,y)) or array-like (length n with 1..n indexing)
    node_coord_raw = data.get("node_coord")
    node_coord: Dict[int, Tuple[float, float]] = {}
    if isinstance(node_coord_raw, dict):
        for idx, val in node_coord_raw.items():
            x, y = val[0], val[1]
            node_coord[int(idx)] = (float(x), float(y))
        dimension = int(data.get("dimension", len(node_coord)))
    else:
        # assume array-like of shape (n,2)
        n = len(node_coord_raw)
        for i in range(1, n + 1):
            xy = node_coord_raw[i - 1]
            node_coord[i] = (float(xy[0]), float(xy[1]))
        dimension = int(data.get("dimension", n))

    # Demands may be dict or array-like
    demand_raw = data.get("demand", {})
    demand_map: Dict[int, int] = {}
    if isinstance(demand_raw, dict):
        for idx, dem in demand_raw.items():
            demand_map[int(idx)] = int(dem)
    else:
        # array-like length n
        n = len(demand_raw)
        for i in range(1, n + 1):
            demand_map[i] = int(demand_raw[i - 1])

    # Reindex so depot is 0, customers 1..n-1
    index_map: Dict[int, int] = {}
    original_ids = list(node_coord.keys())
    original_ids.sort(key=lambda i: (0 if i == depot_original else 1, i))
    for new_idx, old_idx in enumerate(original_ids):
        index_map[old_idx] = new_idx

    coords_arr: List[Tuple[float, float]] = [(0.0, 0.0)] * len(original_ids)
    demands_arr: List[int] = [0] * len(original_ids)
    for old_idx, (x, y) in node_coord.items():
        coords_arr[index_map[old_idx]] = (x, y)
    for old_idx, dem in demand_map.items():
        demands_arr[index_map[old_idx]] = dem if old_idx != depot_original else 0

    dist = _build_distance_matrix(coords_arr)

    return VRPInstance(
        name=name,
        capacity=capacity,
        dimension=len(coords_arr),
        coordinates=coords_arr,
        demands=demands_arr,
        depot_index=0,
        distance_matrix=dist,
    )


def load_solomon_txt_as_cvrp(path: str) -> VRPInstance:
    name = os.path.splitext(os.path.basename(path))[0]
    capacity = 0
    coords: List[Tuple[float, float]] = []
    demands: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]
    n_lines = len(lines)
    # find capacity after VEHICLE block
    v = 0
    while v < n_lines and not lines[v].strip().startswith("VEHICLE"):
        v += 1
    if v < n_lines:
        cap_line_idx = v + 2  # header at v+1, numeric at v+2
        if cap_line_idx < n_lines:
            parts = lines[cap_line_idx].split()
            if len(parts) >= 2:
                try:
                    capacity = int(parts[-1])
                except Exception:
                    try:
                        capacity = int(float(parts[-1]))
                    except Exception:
                        capacity = 0
    # find CUSTOMER header
    i = 0
    while i < n_lines and not lines[i].strip().startswith("CUSTOMER"):
        i += 1
    # move to column header line starting with CUST
    while i < n_lines and not lines[i].strip().startswith("CUST"):
        i += 1
    if i < n_lines:
        i += 1
    # Read rows: idx x y demand ... (ignore time windows)
    raw_rows: List[Tuple[int, float, float, int]] = []
    while i < n_lines:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        parts = line.split()
        if len(parts) < 4:
            i += 1
            continue
        try:
            idx = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            dem = int(float(parts[3]))
        except Exception:
            i += 1
            continue
        raw_rows.append((idx, x, y, dem))
        i += 1

    # Reindex to 0..n-1 with depot id 0 as 0
    raw_rows.sort(key=lambda r: r[0])
    # Expect first row id 0 depot
    coords = []
    demands = []
    for idx, x, y, dem in raw_rows:
        coords.append((x, y))
        demands.append(0 if idx == 0 else dem)

    dist = _build_distance_matrix(coords)
    if len(coords) < 2:
        raise ValueError(f"Parsed zero/one nodes from Solomon file: {path}")
    return VRPInstance(
        name=name,
        capacity=capacity,
        dimension=len(coords),
        coordinates=coords,
        demands=demands,
        depot_index=0,
        distance_matrix=dist,
    )


def load_instance(path: str) -> VRPInstance:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".vrp":
        return load_tsplib_vrp(path)
    if ext == ".txt":
        return load_solomon_txt_as_cvrp(path)
    # fallback: try TSPLIB reader
    return load_tsplib_vrp(path)


