import numpy as np



class Displacement:
    def __init__(self, nu:np.ndarray, s:np.ndarray):
        self.nu = nu
        self.s = s
        self.k2 = np.sqrt(1 - s**2)
        self.k1 = (1 + self.k2**2) ** 2 / (4 * self.k2)
        self.chi = np.sqrt((1 - 2 * self.nu) / (2 - 2 * self.nu))
        self.bb = (
            self.k1 / self.k2 * (1 - self.k2**2)
            + self.k2 / self.k1 * (1 - self.k1**2)
            - (1 - self.k2**4)
        )
        # self.A2 = (1179 - 818 * self.nu + 409 * self.nu**2) / (2100 * (1 - self.nu))
        # self.gama = (2 / (3 * (1 - self.nu))) * (
        #    (7 * self.nu - 17) ** 2 / 200
        # ) - self.A2
        self.gama = (-422 + 424 * self.nu + 33 * self.nu**2) / (1050 * (-1 + self.nu))

    # necessary equations
    def A(self, omega):
        self.omega = omega
        return 0.98 * self.omega / self.s

    def alpha(self):
        return np.sqrt(self.A(self.omega) ** 2 - (self.chi**2) * self.omega**2)

    def beta(self):
        return np.sqrt(self.A(self.omega) ** 2 - self.omega**2)

    # displacements
    def V3RefinedPlate(self):
        return ((3 * (1 - self.nu)) / 4 * (1 + 4 / 5 * self.A(self.omega) ** 2)) / (
            self.A(self.omega) ** 4
            - (3 * (1 - self.nu))
            / 2
            * (1 - (7 * self.nu - 17) / (15 * (1 - self.nu)) * self.A(self.omega) ** 2)
            * self.omega**2
        )

    def V3SimplePlate(self):
        return ((3 * (1 - self.nu)) / 4) / (
            self.A(self.omega) ** 4 - (3 * (1 - self.nu)) / 2 * self.omega**2
        )

    def V3SimpleComposite(self):
        return (
            (
                (3 * (1 - self.nu)) / 4
                + (self.k1 * (1 - self.k2**2)) / (4 * self.bb) * self.A(self.omega) ** 3
            )
        ) / (
            (self.A(self.omega) ** 4)
            - (self.A(self.omega) ** 2 * self.omega**2 / self.s**2)
            - 3 * (1 - self.nu) / 2 * self.omega**2
        )

    def NewV3Composite(self):
        return (
            (3 * (1 - self.nu)) / 4
            + (3 * (1 - self.nu)) / 5 * self.A(self.omega) ** 2
            + self.k1
            * (1 - self.k2**2)
            / 4
            * self.bb
            * self.gama
            * self.A(self.omega) ** 3
            * self.omega**2
        ) / (
            (1 + self.gama * self.omega**2) * self.A(self.omega) ** 4
            - (
                (3 * (1 - self.nu)) / 2
                - (7 * self.nu - 17) / 10 * self.A(self.omega) ** 2
                + self.gama / self.s**2 * self.A(self.omega) ** 2 * self.omega**2
            )
            * self.omega**2
        )

    def V3Rayleigh(self):
        return (self.k1 * (1 - self.k2**2) / (4 * self.bb) * self.A(self.omega)) / (
            self.A(self.omega) ** 2 - (self.omega**2 / self.s**2)
        )

    def V3Exact(self):
        return (
            1
            / 2
            * ((-self.A(self.omega) ** 2 + self.beta() ** 2) * self.alpha())
            / (
                (self.A(self.omega) ** 2 + self.beta() ** 2) ** 2
                * np.tanh(self.alpha())
                - 4
                * self.A(self.omega) ** 2
                * self.alpha()
                * self.beta()
                * np.tanh(self.beta())
            )
        )

    # dispersion relations
    def dispersionrayleigh(self):
        return self.omega / self.s

    def simpledDispersionPlate(self):
        return np.sqrt((3 * (1 - self.nu)) / 2 * self.omega)

    def dispersionPlate(self):
        return np.sqrt(
            -(7 * self.nu - 17) / 20 * self.omega**2
            + 1
            / 2
            * np.sqrt(
                ((7 * self.nu - 17) / 10) ** 2 * self.omega**4
                + 6 * (1 - self.nu) * self.omega**2
            )
        )

    def dispersionComposite(self):
        return np.sqrt(
            (1 / (2 * (1 + self.gama * self.omega**2)))
            * (
                (
                    -(7 * self.nu - 17) / 10 * self.omega**2
                    + self.gama * self.omega**4 / self.s**2
                )
                + np.sqrt(
                    (
                        (7 * self.nu - 17) / 10 * self.omega**2
                        - self.gama * self.omega**4 / self.s**2
                    )
                    ** 2
                    + 6
                    * (1 - self.nu)
                    * (1 + self.gama * self.omega**2)
                    * self.omega**2
                )
            )
        )

    def dispersionExact(self):
        return (
            (self.k2 - self.omega**2 / 2) ** 2 * np.sinh(self.alpha()) / self.alpha()
            - np.cosh(self.beta())
            - self.beta() ** 2
            * self.k2
            * np.cosh(self.alpha())
            * np.sinh(self.beta())
            / self.beta()
        )

    def simpledDispersionComposite(self):
        return np.sqrt(
            1
            / 2
            * (
                self.omega**2 / self.s**2
                + np.sqrt(self.omega**4 / self.s**4 + 6 * (1 - self.nu) * self.omega**2)
            )
        )
