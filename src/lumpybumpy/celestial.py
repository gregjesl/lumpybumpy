import numpy as np

class CelestialBody:
    def __init__(self, name: str, mu: float, radius: float, period: float):
        self.name = name          # Name of the body
        self.mu = mu              # Gravitational parameter (km^3/s^2)
        self.radius = radius      # Mean radius (km)
        self.period = period      # Rotation period (hours)

    def radius_of_altitude(self, h: float) -> float:
        return self.radius + h
    
    def altitude_of_radius(self, r: float) -> float:
        return r - self.radius

    def fixed_to_inertial_dcm(self, time_seconds: float) -> np.ndarray:
        """
        Returns the 3x3 direction cosine matrix (DCM) for the body's rotation
        at a given time since epoch (in seconds).
        """
        # Convert rotation period to radians per second
        omega = 2 * np.pi / (self.period * 3600)  # rad/s

        # Compute rotation angle
        theta = omega * time_seconds

        # Rotation about z-axis
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        return np.array([
            [ cos_t, -sin_t, 0],
            [ sin_t,  cos_t, 0],
            [     0,      0, 1]
        ])

    def inertial_to_fixed_dcm(self, time_seconds: float) -> np.ndarray:
        return self.fixed_to_inertial_dcm(time_seconds).T
    
    def fixed_to_inertial(self, time_seconds: float, r: np.ndarray) -> np.ndarray:
        if len(r) < 2 or len(r) > 3:
            raise ValueError("Unexpected vector length")
        
        return self.fixed_to_inertial_dcm(time_seconds) @ r
    
    def inertial_to_fixed(self, time_seconds: float, r: np.ndarray) -> np.ndarray:
        if len(r) < 2 or len(r) > 3:
            raise ValueError("Unexpected vector length")
        
        return self.inertial_to_fixed_dcm(time_seconds) @ r
    
    def inertial_to_spherical(self, time_seconds: float, r: np.ndarray) -> tuple[float, float, float]:
        """
        Converts an inertial position vector `r` to spherical coordinates (radius, latitude, longitude)
        in the body-fixed frame at `time_seconds` since epoch.
        """
        # Rotate inertial vector into body-fixed frame
        r_fixed = self.inertial_to_fixed(time_seconds, r)

        x, y, z = r_fixed
        radius = float(np.linalg.norm(r_fixed))
        latitude = float(np.arcsin(z / radius))  # radians
        longitude = float(np.arctan2(y, x))      # radians

        return radius, latitude, longitude

# Celestial bodies
SUN     = CelestialBody("Sun",     1.32712e11, 695700.0, 609.12)     # ~25.38 days
MERCURY = CelestialBody("Mercury", 2.20319e4,  2439.4,   1407.6)
VENUS   = CelestialBody("Venus",   3.24859e5,  6051.8,   -5832.5)    # Retrograde
EARTH   = CelestialBody("Earth",   3.98600e5,  6371.0084, 23.9345)
LUNA    = CelestialBody("Luna",    4.90280e3,  1737.4,   655.728)
MARS    = CelestialBody("Mars",    4.28284e4,  3389.5,   24.6229)
JUPITER = CelestialBody("Jupiter", 1.26713e8,  69911.0,  9.925)
SATURN  = CelestialBody("Saturn",  3.79406e7,  58232.0,  10.656)
URANUS  = CelestialBody("Uranus",  5.79456e6,  25362.0, -17.24)      # Retrograde
NEPTUNE = CelestialBody("Neptune", 6.83653e6,  24622.0, 16.11)
PLUTO   = CelestialBody("Pluto",   9.75500e2,  1188.3,   153.2928)