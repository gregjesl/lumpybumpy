import numpy as np

class KeplerianOrbit:
    def __init__(self, mu: float, h: np.ndarray, ecc: np.ndarray, a: float):
        self.mu = mu
        self.h = h
        self.ecc = ecc
        self.a = a

    @staticmethod
    def from_position_velocity(mu: float, r0: np.ndarray, rdot0: np.ndarray):
        h = np.cross(r0, rdot0)
        ecc = (np.cross(rdot0, h) / mu) - (r0 / np.linalg.norm(r0))
        se = (np.dot(rdot0, rdot0) / 2.0) - (mu / np.linalg.norm(r0))
        a = -mu / (2.0 * se)
        orbit = KeplerianOrbit(mu, h, ecc, a)
        sigma = np.dot(r0, rdot0) / np.sqrt(mu)
        ecc_anom = np.arctan2(sigma / np.sqrt(a), 1.0 - (np.linalg.norm(r0) / a))
        return orbit, ecc_anom

    @staticmethod
    def from_elements(mu: float, a: float, e: float, i: float, raan: float, aop: float):
        A11 = np.cos(raan) * np.cos(aop) - np.sin(raan) * np.sin(aop) * np.cos(i)
        A12 = np.sin(raan) * np.cos(aop) + np.cos(raan) * np.sin(aop) * np.cos(i)
        A13 = np.sin(aop) * np.sin(i)
        A31 = np.sin(raan) * np.sin(i)
        A32 = -np.cos(raan) * np.sin(i)
        A33 = np.cos(i)

        ecc = e * np.array([A11, A12, A13])
        rp = a * (1.0 - e)
        vp = np.sqrt((mu / a) * (1.0 + e) / (1.0 - e))
        h_mag = rp * vp
        h = h_mag * np.array([A31, A32, A33])
        return KeplerianOrbit(mu, h, ecc, a)

    @staticmethod
    def from_altitudes(mu: float, rp: float, ra: float, i: float, raan: float, aop: float):
        if rp > ra:
            raise ValueError("Periapsis is larger than apoapsis")
        a = (rp + ra) / 2.0
        e = (ra / a) - 1.0
        return KeplerianOrbit.from_elements(mu, a, e, i, raan, aop)

    def r_mag(self, ecc_anom: float):
        return self.a * (1.0 - self.e() * np.cos(ecc_anom))

    def in_plane_position(self, ecc_anom: float) -> np.ndarray:
        x = self.a * (np.cos(ecc_anom) - self.e())
        y = self.a * np.sqrt(1 - self.e()**2) * np.sin(ecc_anom)
        return np.array([x, y, 0.0])

    def in_plane_velocity(self, ecc_anom: float) -> np.ndarray:
        r = self.r_mag(ecc_anom)
        vx = -np.sqrt(self.mu * self.a) * np.sin(ecc_anom) / r
        vy = np.sqrt(self.mu * self.a * (1.0 - self.e()**2)) * np.cos(ecc_anom) / r
        return np.array([vx, vy, 0.0])

    def to_position_velocity(self, ecc_anom: float):
        dcm = self.dcm().T
        pos = dcm @ self.in_plane_position(ecc_anom)
        vel = dcm @ self.in_plane_velocity(ecc_anom)
        return pos, vel

    def sma(self):
        return self.a

    def semi_latus_rectum(self):
        return self.a * (1.0 - np.linalg.norm(self.ecc)**2)

    def e(self):
        return np.linalg.norm(self.ecc)

    def dcm(self):
        ie = self.ecc / np.linalg.norm(self.ecc)
        ih = self.h / np.linalg.norm(self.h)
        j = np.cross(ih, ie)
        return np.column_stack((ie, j, ih))

    def inc(self):
        return np.arccos(self.dcm()[2, 2])

    def raan(self):
        dcm = self.dcm()
        result = np.arctan2(dcm[2, 0], -dcm[2, 1])
        return result if result >= 0 else result + 2.0 * np.pi

    def arg_periapsis(self):
        dcm = self.dcm()
        result = np.arctan2(dcm[0, 2], dcm[1, 2])
        return result if result >= 0 else result + 2.0 * np.pi

    def mean_motion(self):
        return np.sqrt(self.mu / self.a**3)

    def period(self):
        return 2.0 * np.pi / self.mean_motion()