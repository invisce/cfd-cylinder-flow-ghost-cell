"""
Defect Geometry Module for Corroded / Damaged Cylinder Wakes.

Implements the localized super-Gaussian radial recess defined in the research plan:
    r(theta) = R - delta * exp( - ((theta - theta_d) / phi)^4 )

Parameters:
    R       : Base cylinder radius (D/2)
    delta   : Defect depth (delta = d * D)
    theta_d : Angular center of defect (radians, 0 at upstream stagnation point)
    phi     : Angular half-width (radians, w = 2*phi)
"""

import numpy as np


def wrap_angle(theta):
    """Wrap angle to [-pi, pi]."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


class DefectCylinderGeometry:
    """
    Mathematical representation of a 2D cylinder with a localized super-Gaussian defect.
    
    Coordinate convention:
        - theta = 0 at upstream stagnation point (-x direction relative to center)
          or standard Cartesian angle if specified. Here theta is measured in standard
          Cartesian angle: x = xc + r*cos(theta), y = yc + r*sin(theta).
        - Upstream stagnation point in flow with U_inf in +x direction is theta = pi.
        - To conform with professor specification where theta_d = 0 at stagnation point:
          we support stagnation-referenced angle: theta_flow = theta - pi.
    """

    def __init__(self, xc=0.0, yc=0.0, D=1.0, depth_ratio=0.0, width_rad=0.0, theta_d_rad=0.0, stagnation_at_pi=True):
        """
        Parameters:
            xc, yc           : Center coordinates of cylinder.
            D                : Nominal cylinder diameter (R = D / 2).
            depth_ratio      : Non-dimensional depth d = delta / D in [0, 0.10].
            width_rad        : Angular width w = 2*phi in radians.
            theta_d_rad      : Center location of defect in radians relative to stagnation point.
            stagnation_at_pi : If True, stagnation point is at Cartesian theta = pi (flow in +x).
        """
        self.xc = float(xc)
        self.yc = float(yc)
        self.D = float(D)
        self.R = float(D) / 2.0
        self.d = float(depth_ratio)
        self.delta = self.d * self.D
        self.w = float(width_rad)
        self.phi = self.w / 2.0 if self.w > 0 else 1.0
        self.theta_d_rel = float(theta_d_rad)
        self.stagnation_at_pi = bool(stagnation_at_pi)
        
        # Absolute Cartesian angle of defect center
        if self.stagnation_at_pi:
            self.theta_d = wrap_angle(np.pi + self.theta_d_rel)
        else:
            self.theta_d = wrap_angle(self.theta_d_rel)

    def radius(self, theta):
        """
        Compute radial distance r(theta) from cylinder center at angle(s) theta.
        """
        if self.delta <= 1e-15 or self.phi <= 1e-15:
            if np.isscalar(theta):
                return self.R
            return np.full_like(theta, self.R, dtype=float)

        d_theta = wrap_angle(theta - self.theta_d)
        xi = d_theta / self.phi
        # Super-Gaussian exponent ((Delta theta) / phi)^4
        # Clamp to avoid underflow/overflow in exponent
        xi4 = np.clip(xi**4, 0.0, 100.0)
        recess = self.delta * np.exp(-xi4)
        return self.R - recess

    def dr_dtheta(self, theta):
        """
        Analytical first derivative dr/dtheta.
        dr/dtheta = delta * (4 / phi) * (Delta theta / phi)^3 * exp( - (Delta theta / phi)^4 )
        """
        if self.delta <= 1e-15 or self.phi <= 1e-15:
            if np.isscalar(theta):
                return 0.0
            return np.zeros_like(theta, dtype=float)

        d_theta = wrap_angle(theta - self.theta_d)
        xi = d_theta / self.phi
        xi3 = xi**3
        xi4 = np.clip(xi**4, 0.0, 100.0)
        return self.delta * (4.0 / self.phi) * xi3 * np.exp(-xi4)

    def surface_point(self, theta):
        """
        Compute Cartesian coordinates (x, y) on the defect cylinder surface.
        """
        r = self.radius(theta)
        x = self.xc + r * np.cos(theta)
        y = self.yc + r * np.sin(theta)
        return x, y

    def unit_normal(self, theta):
        """
        Compute outward unit normal vector (nx, ny) on the physical surface.
        
        Tangent vector t = dr/dtheta:
            tx = r'*cos(theta) - r*sin(theta)
            ty = r'*sin(theta) + r*cos(theta)
        Outward normal n (rotated 90 deg clockwise relative to t along increasing theta):
            nx = (r*cos(theta) + r'*sin(theta)) / sqrt(r^2 + (r')^2)
            ny = (r*sin(theta) - r'*cos(theta)) / sqrt(r^2 + (r')^2)
        """
        r = self.radius(theta)
        rp = self.dr_dtheta(theta)
        
        nx_unnorm = r * np.cos(theta) + rp * np.sin(theta)
        ny_unnorm = r * np.sin(theta) - rp * np.cos(theta)
        
        norm = np.hypot(nx_unnorm, ny_unnorm)
        # Avoid division by zero
        norm = np.maximum(norm, 1e-14)
        return nx_unnorm / norm, ny_unnorm / norm

    def arc_length_element(self, theta):
        """
        Differential arc length element ds/dtheta = sqrt(r^2 + (r')^2).
        """
        r = self.radius(theta)
        rp = self.dr_dtheta(theta)
        return np.hypot(r, rp)

    def signed_distance(self, x, y):
        """
        Compute approximate / high-order Signed Distance Function (SDF) at point(s) (x, y).
        Convention:
            SDF < 0 inside cylinder (solid)
            SDF = 0 on boundary
            SDF > 0 in fluid
            
        Uses radial distance corrected by local surface curvature / normal projection.
        For smooth shallow defects (delta/R <= 0.20), the projection along ray from center
        gives an accurate approximation:
            dist ~ sqrt((x-xc)^2 + (y-yc)^2) - r(theta)
        Refined with 1st-order normal correction:
            dist_true ~ (r_pt - r_surf) * (r_surf / sqrt(r_surf^2 + (r')^2))
        """
        dx = x - self.xc
        dy = y - self.yc
        r_pt = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx)
        
        r_surf = self.radius(theta)
        rp = self.dr_dtheta(theta)
        
        # Ray distance
        ray_dist = r_pt - r_surf
        
        # Scaling projection onto outward normal at ray intersection
        # cos(alpha) = r_surf / sqrt(r_surf^2 + rp^2)
        cos_alpha = r_surf / np.maximum(np.hypot(r_surf, rp), 1e-14)
        return ray_dist * cos_alpha
