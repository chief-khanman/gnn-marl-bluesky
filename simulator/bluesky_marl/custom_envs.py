"""Configurable subclass of the bluesky-gym `SectorCR` environment (goals G04.1, G06.1; S-06).

The submodule is not edited. This subclass changes two things, both decided on 2026-11-25
(D-028, D-029; plan docs/plans/G05.3_gnn_qmix.md §3.7):

  * `spawn_min_sep_nm`: no two aircraft are created closer than this distance, so no pair is
    already in loss of separation at the first step (GAP-25). 0 keeps the original spawning.
  * `area_scale`: multiplies the sector area range (2400-3750 NM^2 in the original). 1.0 is
    the original "fixed" airspace, where traffic density rises with the fleet size.
    `area_scale = n / n_ref` keeps the density of a reference fleet size ("density" airspace, GAP-19).

`_generate_polygon` is copied from bluesky_zoo/sector_cr/sector_cr.py @ 3a536c0 with one
change: the area range is `self.area_range` instead of the module constant POLY_AREA_RANGE.
"""
from __future__ import annotations

import bluesky as bs
import numpy as np
from bluesky_zoo.sector_cr import sector_cr as _base

import bluesky_gym.envs.common.functions as fn

MAX_TRIES_PER_AIRCRAFT = 2000
MAX_LAYOUTS = 500


class ConfigurableSectorCR(_base.SectorCR):
    def __init__(self, render_mode=None, n_agents=10, spawn_min_sep_nm: float = 0.0, area_scale: float = 1.0):
        if area_scale <= 0:
            raise ValueError("area_scale must be positive")
        self.spawn_min_sep_nm = float(spawn_min_sep_nm)
        self.area_range = tuple(a * float(area_scale) for a in _base.POLY_AREA_RANGE)
        super().__init__(render_mode=render_mode, n_agents=n_agents)

    def _generate_polygon(self):
        R = np.sqrt(self.area_range[1] / np.pi)
        p = [fn.random_point_on_circle(R) for _ in range(3)]
        p = fn.sort_points_clockwise(p)
        p_area = fn.polygon_area(p)
        while p_area < self.area_range[0]:
            p.append(fn.random_point_on_circle(R))
            p = fn.sort_points_clockwise(p)
            p_area = fn.polygon_area(p)
        self.poly_area = p_area
        self.poly_points = np.array(p)  # NM
        p = [fn.nm_to_latlong(_base.CENTER, point) for point in p]
        points = [coord for point in p for coord in point]
        bs.tools.areafilter.defineArea(self.poly_name, "POLY", points)

    def _sample_positions_nm(self) -> list[np.ndarray] | None:
        """Sequential rejection sampling inside the sector. None if the layout cannot be completed."""
        lo, hi = self.poly_points.min(axis=0), self.poly_points.max(axis=0)
        alt_m = np.array([_base.ALTITUDE * _base.FL2M])
        placed: list[np.ndarray] = []
        for _ in range(self.num_ac):
            for _ in range(MAX_TRIES_PER_AIRCRAFT):
                p = np.array([np.random.uniform(lo[0], hi[0]), np.random.uniform(lo[1], hi[1])])
                ll = fn.nm_to_latlong(_base.CENTER, p)
                if not bs.tools.areafilter.checkInside(self.poly_name, np.array([ll[0]]), np.array([ll[1]]), alt_m):
                    continue
                if all(np.hypot(*(p - q)) >= self.spawn_min_sep_nm for q in placed):
                    placed.append(p)
                    break
            else:
                return None
        return placed

    def _generate_ac(self) -> None:
        if self.spawn_min_sep_nm <= 0:
            return super()._generate_ac()
        positions = self._sample_positions_nm()
        layouts = 1
        while positions is None:  # this sector is too small for the fleet: draw another sector
            if layouts >= MAX_LAYOUTS:
                raise RuntimeError(
                    f"could not place {self.num_ac} aircraft {self.spawn_min_sep_nm} NM apart in a sector of "
                    f"{self.area_range[0]:.0f}-{self.area_range[1]:.0f} NM^2")
            layouts += 1
            bs.tools.areafilter.deleteArea(self.poly_name)
            self._generate_polygon()
            self._generate_waypoints()
            positions = self._sample_positions_nm()
        for idx, agent in enumerate(self.agents):
            wpt = fn.nm_to_latlong(_base.CENTER, self.wpts[idx])
            pos = fn.nm_to_latlong(_base.CENTER, positions[idx])
            bs.traf.cre(agent, actype=_base.AC_TYPE, aclat=pos[0], aclon=pos[1], achdg=fn.get_hdg(pos, wpt),
                        acspd=_base.AC_SPD, acalt=_base.ALTITUDE)
