import os
import math
import numpy as np
import re

class CVRPInstance:
    """
    Handles reading and storing data for a Capacitated Vehicle Routing Problem (CVRP)
    instance. Automatically supports:
      - Augerat .vrp files (NODE_COORD_SECTION, DEMAND_SECTION, DEPOT_SECTION)
      - Solomon .txt files (VEHICLE / CUSTOMER sections)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.customers = []
        self.vehicle_capacity = None
        self.num_vehicles = None
        self.dist_matrix = None
        self.demands = []

        self._parse_file()
        self._build_distance_matrix()

    def _parse_file(self):
        with open(self.filepath, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        # --- Detect file type ---
        if any("NODE_COORD_SECTION" in ln.upper() for ln in lines):
            self._parse_vrp_format(lines)
        elif any("CUSTOMER" in ln.upper() for ln in lines):
            self._parse_txt_format(lines)
        else:
            raise ValueError(f"Unrecognized VRP format in {self.filepath}")

    # ---------------------------------------------------------------------
    # A-n32-k5.vrp (Augerat-style)
    # ---------------------------------------------------------------------
    def _parse_vrp_format(self, lines):

        coords = {}
        demands = {}
        depot_id = None

        # --- Read meta info ---
        for line in lines:
            if line.startswith("CAPACITY") or "CAPACITY" in line:
                self.vehicle_capacity = int(line.split(":")[-1])
            if line.startswith("COMMENT") and "VEHICLES" in line.upper():
                # Sometimes num vehicles is mentioned in the comment
                m = re.search(r"VEHICLES\s*:\s*(\d+)", line.upper())
                if m:
                    self.num_vehicles = int(m.group(1))

        # --- Fallback: extract kXX from filename if not found ---
        if self.num_vehicles is None:
            fname = os.path.basename(self.filepath)
            m = re.search(r'k(\d+)', fname.lower())
            if m:
                self.num_vehicles = int(m.group(1))

        # --- Sections ---
        node_section = lines.index(next(l for l in lines if "NODE_COORD_SECTION" in l.upper())) + 1
        demand_section = lines.index(next(l for l in lines if "DEMAND_SECTION" in l.upper())) + 1
        depot_section = lines.index(next(l for l in lines if "DEPOT_SECTION" in l.upper())) + 1

        # --- Node coordinates ---
        for line in lines[node_section:]:
            if "DEMAND_SECTION" in line.upper():
                break
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))

        # --- Demands ---
        for line in lines[demand_section:]:
            if "DEPOT_SECTION" in line.upper():
                break
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                demands[int(parts[0])] = int(parts[1])

        # --- Depot ---
        for line in lines[depot_section:]:
            if line.strip() == "-1":
                break
            if line.strip().isdigit():
                depot_id = int(line.strip())

        # --- Combine ---
        self.customers = []
        for cid in sorted(coords.keys()):
            self.customers.append({
                "id": cid,
                "x": coords[cid][0],
                "y": coords[cid][1],
                "demand": demands.get(cid, 0)
            })

        # Ensure depot (id 1 in most cases) is first
        self.customers.sort(key=lambda c: (c["id"] != depot_id, c["id"]))
        self.demands = [c["demand"] for c in self.customers]


    # ---------------------------------------------------------------------
    # C101.txt (Solomon / CVRP-style)
    # ---------------------------------------------------------------------
    def _parse_txt_format(self, lines):
        # Find VEHICLE info
        for i, line in enumerate(lines):
            if line.upper().startswith("VEHICLE"):
                j = i + 1
                while j < len(lines):
                    parts = lines[j].split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        self.num_vehicles = int(parts[0])
                        self.vehicle_capacity = int(parts[1])
                        break
                    j += 1
                break

        # Find CUSTOMER section
        cust_start = None
        for i, line in enumerate(lines):
            if line.upper().startswith("CUSTOMER"):
                cust_start = i + 1
                break

        # Extract customers
        self.customers = []
        for line in lines[cust_start:]:
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                cid = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = int(parts[3])
                self.customers.append({
                    "id": cid,
                    "x": x,
                    "y": y,
                    "demand": demand
                })

        self.customers.sort(key=lambda c: c["id"])
        self.demands = [c["demand"] for c in self.customers]

    # ---------------------------------------------------------------------
    def _build_distance_matrix(self):
        n = len(self.customers)
        self.dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.dist_matrix[i, j] = math.hypot(
                        self.customers[i]["x"] - self.customers[j]["x"],
                        self.customers[i]["y"] - self.customers[j]["y"]
                    )

    def summary(self):
        print(f"\nInstance: {os.path.basename(self.filepath)}")
        print(f"Vehicles: {self.num_vehicles or 'N/A'} | Capacity: {self.vehicle_capacity}")
        print(f"Customers (inc. depot): {len(self.customers)}")

