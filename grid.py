from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

from types_common import GridPos


class GridMap:
    """
    GridMap khusus warehouse.

    Kode cell:
        0: empty space (navigable path)
        1: walls/obstacles (black)
        2: shelves (blue)               -> obstacles
        3: loading stations (green)
        4: docking stations (yellow)
        5: narrow aisles (papayawhip)   -> dianggap navigable
        6: intersections (lightgray)    -> dianggap navigable

    Untuk MGCBS:
        - hanya cell dengan obstacles[y,x] == True yang tidak bisa dilalui
        - sehingga narrow aisle & intersection = empty space (hanya dibuat sebagai visualisasi).
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)

        # obstacles[y,x] = True jika tidak bisa dilalui (wall/shelf)
        self.obstacles = np.zeros((height, width), dtype=bool)

    # ------------------------------------------------------------------
    # Builder layout
    # ------------------------------------------------------------------
    def add_walls(self, walls_list: List[GridPos]) -> None:
        for x, y in walls_list:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 1
                self.obstacles[y, x] = True

    def add_shelves(self, shelves_list: List[GridPos]) -> None:
        for x, y in shelves_list:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 2
                self.obstacles[y, x] = True

    def add_stations(self, stations_list: List[GridPos]) -> None:
        for x, y in stations_list:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 3

    def add_dockings(self, dockings_list: List[GridPos]) -> None:
        for x, y in dockings_list:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.grid[y, x] = 4

    def add_narrowaisles(self, narrowaisles_list: List[GridPos]) -> None:
        for x, y in narrowaisles_list:
            if 0 <= x < self.width and 0 <= y < self.height:
                # hanya label visual; tidak jadi obstacle
                self.grid[y, x] = 5

    def add_intersections(self, intersections_list: List[GridPos]) -> None:
        for x, y in intersections_list:
            if 0 <= x < self.width and 0 <= y < self.height:
                # hanya label visual; tidak jadi obstacle
                self.grid[y, x] = 6

    # ------------------------------------------------------------------
    # Fungsi dasar yang dipakai MGCBS
    # ------------------------------------------------------------------
    def in_bounds(self, pos: GridPos) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, pos: GridPos) -> bool:
        x, y = pos
        if not self.in_bounds(pos):
            return False
        return not self.obstacles[y, x]

    def neighbors(self, pos: GridPos) -> List[GridPos]:
        """4-neighbor (tanpa aksi 'stay'); MGCBS sudah punya aksi wait di level waktu."""
        x, y = pos
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        results: List[GridPos] = []
        for nx, ny in candidates:
            np = (nx, ny)
            if self.in_bounds(np) and self.passable(np):
                results.append(np)
        return results
    
    # ------------------------------------------------------------------
    # Helper query
    # ------------------------------------------------------------------
    def is_obstacle(self, x: int, y: int) -> bool:
        if not self.in_bounds((x, y)):
            return True
        return self.obstacles[y, x]

    def get_empty_cells(self) -> List[GridPos]:
        cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] == 0:
                    cells.append((x, y))
        return cells

    def get_walls(self) -> List[GridPos]:
        return [(x, y) for y in range(self.height) for x in range(self.width)
                if self.grid[y, x] == 1]

    def get_shelves(self) -> List[GridPos]:
        return [(x, y) for y in range(self.height) for x in range(self.width)
                if self.grid[y, x] == 2]

    def get_stations(self) -> List[GridPos]:
        return [(x, y) for y in range(self.height) for x in range(self.width)
                if self.grid[y, x] == 3]

    def get_docking_stations(self) -> List[GridPos]:
        return [(x, y) for y in range(self.height) for x in range(self.width)
                if self.grid[y, x] == 4]

    # ------------------------------------------------------------------
    # Visualisasi
    # ------------------------------------------------------------------
    def visualize(self) -> None:
        # 0,1,2,3,4,5,6 -> warna
        colors = [
            'white',     # 0 empty
            'black',     # 1 wall
            'blue',      # 2 shelf
            'green',     # 3 loading station
            'yellow',    # 4 docking
            'papayawhip',# 5 narrow aisle
            'lightgray', # 6 intersection
        ]
        cmap = ListedColormap(colors)

        plt.figure(figsize=(15, 6))
        plt.imshow(self.grid, cmap=cmap, origin="lower")

        # grid lines
        for x in range(self.width + 1):
            plt.axvline(x - 0.5, color='gray', linewidth=0.5)
        for y in range(self.height + 1):
            plt.axhline(y - 0.5, color='gray', linewidth=0.5)

        legend_elements = [
            patches.Patch(facecolor='white', edgecolor='gray', label='Empty Space'),
            patches.Patch(facecolor='black', label='Wall/Obstacle'),
            patches.Patch(facecolor='blue', label='Shelf'),
            patches.Patch(facecolor='green', label='Loading Station'),
            patches.Patch(facecolor='yellow', label='Docking Station'),
            patches.Patch(facecolor='papayawhip', label='Narrow Aisle'),
            patches.Patch(facecolor='lightgray', label='Intersection'),
        ]
        plt.legend(handles=legend_elements,
                   loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=4)

        plt.title('Warehouse Layout')
        plt.tight_layout()
        plt.show()

