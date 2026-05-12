import numpy as np

class Material:

    # slip systems for FCC crystal
    s_a = [
        np.array([0.0, 1.0, -1.0]),
        np.array([-1.0, 0.0, 1.0]),
        np.array([1.0, -1.0, 0.0]),
        np.array([0.0, 1.0, -1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([-1.0, -1.0, 0.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([-1.0, -1.0, 0.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
        -np.array([0.0, 1.0, -1.0]),
        -np.array([-1.0, 0.0, 1.0]),
        -np.array([1.0, -1.0, 0.0]),
        -np.array([0.0, 1.0, -1.0]),
        -np.array([1.0, 0.0, 1.0]),
        -np.array([-1.0, -1.0, 0.0]),
        -np.array([0.0, 1.0, 1.0]),
        -np.array([1.0, 0.0, -1.0]),
        -np.array([-1.0, -1.0, 0.0]),
        -np.array([0.0, 1.0, 1.0]),
        -np.array([1.0, -1.0, 0.0]),
        -np.array([1.0, 0.0, 1.0]),
    ]

    n_a = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([1.0, 1.0, -1.0]),
        np.array([1.0, 1.0, -1.0]),
        np.array([1.0, 1.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, 1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([-1.0, 1.0, -1.0]),
        np.array([1.0, 1.0, -1.0]),
        np.array([1.0, 1.0, -1.0]),
        np.array([1.0, 1.0, -1.0]),
    ]

    def __init__(self, E=60_888, nu = 0.3, tau0 = 60.84, q = 1.0, xi = 541.48,
    tau_inf=109.51, phi = (0.0, 0.0, 0.0) ):
        """Initialize material parameters. Default parameters acc. to Scheunemann2017 Tab. 8.1 Al-Cu alloy.
        """

        # Material Parameters
        self.E = E
        self.nu = nu
        self.lame_lambda = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        self.lame_mu = self.E / (2 * (1 + self.nu))
        self.C = self._compute_elasticity_tensor()

        # Hardening related parameters
        self.tau0 = tau0
        self.xi = xi
        self.tau_inf = tau_inf
        self.q = q

        # crystal plasticity related parameters
        self.phi = phi # Euler angles
        self.Za = self._compute_projection_tensors()

    def _compute_elasticity_tensor(self) -> np.ndarray:
        """
        Compute the 4th order elasticity tensor C for isotropic materials.

        :return: elasticity tensor C as a 3x3x3x3 numpy array
        """
        C = np.zeros((3, 3, 3, 3))
        lam = self.lame_lambda
        mu = self.lame_mu

        I = np.eye(3)

        # C_ijkl = lambda * delta_ij * delta_kl + mu * (delta_ik * delta_jl + delta_il * delta_jk)
        C = lam * np.einsum('ij,kl->ijkl', I, I) + mu * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))

        return C

    @staticmethod
    def rotation_from_euler(phi: tuple) -> np.ndarray:
        """Build a 3x3 rotation matrix from Euler angles (phi1, phi2, phi3)."""
        phi1, phi2, phi3 = phi
        theta1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi1), -np.sin(phi1)],
            [0.0, np.sin(phi1), np.cos(phi1)]
        ])
        theta2 = np.array([
            [np.cos(phi2), 0.0, np.sin(phi2)],
            [0.0, 1.0, 0.0],
            [-np.sin(phi2), 0.0, np.cos(phi2)]
        ])
        theta3 = np.array([
            [np.cos(phi3), -np.sin(phi3), 0.0],
            [np.sin(phi3), np.cos(phi3), 0.0],
            [0.0, 0.0, 1.0]
        ])
        return theta3 @ theta2 @ theta1

    def _compute_projection_tensors(self, rot: np.ndarray | None = None) -> list[np.ndarray]:
        """
        Compute the projection tensors.

        :param rot: optional 3x3 rotation matrix; if None, built from self.phi
        :return: list of 3x3 numpy arrays representing projection tensors
        """
        if rot is None:
            rot = self.rotation_from_euler(self.phi)

        Zalpha = []
        for s, n in zip(self.s_a, self.n_a):
            # Normalize vectors
            s = s / np.linalg.norm(s)
            n = n / np.linalg.norm(n)

            # Rotate vectors
            sr = rot @ s
            nr = rot @ n

            # Symmetric outer product: (m ⊗ n + n ⊗ m) / 2
            Z = 0.5 * (np.outer(sr, nr) + np.outer(nr, sr))
            Zalpha.append(Z)

        return Zalpha

    def set_orientation(self, rot: np.ndarray) -> None:
        """Recompute projection tensors from a given 3x3 rotation matrix."""
        self.Za = self._compute_projection_tensors(rot)

    def adjust_projection_tensors(self, sigma: np.ndarray) -> None:
        """
        Adjust the projection tensors based on the Cauchy stress sigma such that
        only positive resolved shear stresses are considered. Only necessary for
        implementations using 12 instead of 24 slip systems.

        :param sigma: Cauchy stress tensor
        """
        for i,Za in enumerate(self.Za):
            if np.einsum('ij,ij->', sigma, Za) < 0:
                self.Pa[i] = -Za

    def print_material_parameters(self) -> None:
        """ Print the material parameters for verification.
        """
        print("== Material Parameters:")
        print(f"E: {self.E}")
        print(f"nu: {self.nu}")
        print(f"lambda (1st Lamé): {self.lame_lambda}")
        print(f"mu (2nd Lamé): {self.lame_mu}")

        print("== Hardening Parameters:")
        print(f"tau0: {self.tau0}")
        print(f"tau_inf: {self.tau_inf}")
        print(f"xi: {self.xi}")
        print(f"q: {self.q}")

        print("== Crystal Parameters:")
        print(f"phi: {self.phi}")


    def initialize_history(self) -> dict:
        """
        Initialize the material history dictionary.

        :return: history dictionary with initial values
        """
        nSlip = len(self.Za)
        hist = {
            "eps": np.zeros((3, 3)),
            "eps_p": np.zeros((3, 3)),
            "gamma_a": np.zeros(nSlip),
            "tau_h": np.ones(nSlip) * self.tau0,
        }
        return hist

def gamma24_to_12(gamma_a: np.ndarray) -> np.ndarray:
    """Accumulate 24 slip values to 12 physical slip systems.

    Systems 13..24 share the same slip plane as 1..12 but with
    opposite slip direction.  The net slip on each physical system
    is gamma^alpha + gamma^{alpha+12}.

    :param gamma_a: (24,) array of accumulated slips
    :return: (12,) array of net slips per physical slip system
    """
    return gamma_a[:12] + gamma_a[12:]

def ten2voigt(A: np.ndarray, fact: int = 2) -> np.ndarray:
    return np.array([A[0,0],A[1,1],A[2,2], fact*A[0,1], fact*A[0,2],fact*A[1,2]])

def voigt2ten(a: np.ndarray, fact: int = 2) -> np.ndarray:
    return np.array([ [a[0],a[3]/fact,a[4]/fact],
                      [a[3]/fact,a[1],a[5]/fact],
                      [a[4]/fact,a[5]/fact,a[2]]])
