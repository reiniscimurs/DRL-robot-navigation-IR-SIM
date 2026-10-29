"""Open waypoint splines with distances measured along the path in world units."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


class SplineTrajectory:
    """Interpolate ordered (x, y) waypoints using a natural cubic spline.

    Arc length is approximated by a dense chord table. Increase
    ``samples_per_segment`` for tightly curved paths. Paths are open; distances
    outside the path clamp to its endpoints. Cubic interpolation can overshoot
    the waypoint polyline, so waypoints do not imply obstacle clearance.
    """

    def __init__(self, waypoints, samples_per_segment=100):
        points = np.asarray(waypoints, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.all(np.isfinite(points))):
            raise ValueError("waypoints must contain at least two finite (x, y) points")
        if not isinstance(samples_per_segment, (int, np.integer)) or samples_per_segment < 2:
            raise ValueError("samples_per_segment must be an integer >= 2")
        chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(chords <= 0):
            raise ValueError("consecutive waypoints must be distinct")
        knots = np.r_[0.0, np.cumsum(chords)]
        self._spline = CubicSpline(knots, points, axis=0, bc_type="natural")
        self._u = np.concatenate([
            np.linspace(a, b, samples_per_segment, endpoint=False)
            for a, b in zip(knots[:-1], knots[1:])
        ] + [knots[-1:]])
        self._points = self._spline(self._u)
        self._arc = np.r_[0.0, np.cumsum(
            np.linalg.norm(np.diff(self._points, axis=0), axis=1))]
        self.length = float(self._arc[-1])

    def point_at(self, distance):
        """Evaluate one or many arc distances, clamped to [0, length]."""
        return self._spline(np.interp(distance, self._arc, self._u))

    def sample_points(self):
        """Return a copy of the dense (N, 2) spline samples for visualization."""
        return self._points.copy()

    def heading_at(self, distance):
        tangent = self._spline(np.interp(distance, self._arc, self._u), 1)
        return np.arctan2(tangent[..., 1], tangent[..., 0])

    def project(self, point):
        """Return (nearest spline point, arc distance).

        Refine each sampled local distance minimum, then choose the closest.
        Resolution controls detection of very small features. At crossings the
        nearest branch wins; this method does not track route history.
        """
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("point must be a finite (x, y) pair")
        distances = np.sum((self._points - point) ** 2, axis=1)
        minima = np.flatnonzero((distances[1:-1] <= distances[:-2])
                                & (distances[1:-1] <= distances[2:])) + 1
        candidates = [self._u[0], self._u[-1]]
        for i in minima:
            result = minimize_scalar(
                lambda u: np.sum((self._spline(u) - point) ** 2),
                bounds=(self._u[i - 1], self._u[i + 1]), method="bounded",
                options={"xatol": 1e-10},
            )
            candidates.append(result.x)
        u = min(candidates, key=lambda u: np.sum((self._spline(u) - point) ** 2))
        return self._spline(u), float(np.interp(u, self._u, self._arc))

    def lookahead_points(self, distance, offsets):
        """Return points at nonnegative arc offsets ahead of ``distance``."""
        offsets = np.asarray(offsets, dtype=float)
        if offsets.ndim != 1 or not np.all(np.isfinite(offsets)) or np.any(offsets < 0):
            raise ValueError("lookahead distances must be a finite nonnegative sequence")
        return self.point_at(distance + offsets)
