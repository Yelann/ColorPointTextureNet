<<<<<<< Updated upstream
from typing import Tuple
import robust_laplacian
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

def compute_eig_laplacian(
    verts: np.ndarray,
    faces: np.ndarray | None,
    k_eig: int = 10,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, scipy.sparse.csc_matrix, np.ndarray]:
    """Compute the eigendecomposition of the Laplacian

    Args:
        verts (np.ndarray): [N x 3] N vertices of the mesh
        faces (np.ndarray | None): [F x 3] F triangular faces of the mesh
        k_eig (int, optional): number of eigenvalues and eigenvectors desired.
            Defaults to 10.
        eps (float, optional): constant used to perturb Laplacian during
            eigendecomposition. Defaults to 1e-8.

    Raises:
        ValueError: although multiple attempts were made, the eigendecomposition
            failed.

    Returns:
        Tuple[np.ndarray, np.ndarray, scipy.sparse.csc_matrix, np.ndarray]:
            k eigenvalues, [k x N] eigenvectors, [N x N] Laplacian, and
            [N] mass vector
    """
    lapl, mass = robust_laplacian.mesh_laplacian(verts, faces)
    massvec = mass.diagonal()

    # Prepare matrices for eigendecomposition like in DiffusionNet code
    lapl_eigsh = (lapl + scipy.sparse.identity(lapl.shape[0]) * eps).tocsc()
    mass_mat = scipy.sparse.diags(massvec)
    eigs_sigma = eps

    failcount = 0
    while True:
        try:
            evals, evecs = scipy.sparse.linalg.eigsh(
                lapl_eigsh, k=k_eig, M=mass_mat, sigma=eigs_sigma
            )
            evals = np.clip(evals, a_min=0.0, a_max=float("inf"))
            break
        except RuntimeError as exc:
            if failcount > 2:
                print("Fail", failcount)
                # raise ValueError("failed to compute eigendecomp") from exc
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            lapl_eigsh = lapl_eigsh + scipy.sparse.identity(lapl.shape[0]) * (
                eps * 10**failcount
            )
    return evals, evecs, lapl, massvec


def compute_tot_area(pos: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    side_1 = pos[faces[1]] - pos[faces[0]]
    side_2 = pos[faces[2]] - pos[faces[0]]
    return side_1.cross(side_2).norm(p=2, dim=1).abs().sum() / 2


def get_shape_condition(vertices, faces, dim):
    evals, evecs, lapl, massvec = compute_eig_laplacian(vertices, faces, dim)
    surf_area = compute_tot_area(torch.from_numpy(vertices), torch.from_numpy(faces))
    # Evals change with scale, follow first steps of cShapeDNA to get them
    # on a straight line and remove scale dependency.
    evals = torch.from_numpy(evals)
    arange = torch.arange(0, dim).float()
    # arange = torch.arange(0, evals.shape[1]).float()
    shape_desc = evals * surf_area - 4 * torch.pi * arange
    # shape_desc = evals * surf_area[:, None] - 4 * torch.pi * arange
=======
from typing import Tuple
import robust_laplacian
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

def compute_eig_laplacian(
    verts: np.ndarray,
    faces: np.ndarray | None,
    k_eig: int = 10,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, scipy.sparse.csc_matrix, np.ndarray]:
    """Compute the eigendecomposition of the Laplacian

    Args:
        verts (np.ndarray): [N x 3] N vertices of the mesh
        faces (np.ndarray | None): [F x 3] F triangular faces of the mesh
        k_eig (int, optional): number of eigenvalues and eigenvectors desired.
            Defaults to 10.
        eps (float, optional): constant used to perturb Laplacian during
            eigendecomposition. Defaults to 1e-8.

    Raises:
        ValueError: although multiple attempts were made, the eigendecomposition
            failed.

    Returns:
        Tuple[np.ndarray, np.ndarray, scipy.sparse.csc_matrix, np.ndarray]:
            k eigenvalues, [k x N] eigenvectors, [N x N] Laplacian, and
            [N] mass vector
    """
    lapl, mass = robust_laplacian.mesh_laplacian(verts, faces)
    massvec = mass.diagonal()

    # Prepare matrices for eigendecomposition like in DiffusionNet code
    lapl_eigsh = (lapl + scipy.sparse.identity(lapl.shape[0]) * eps).tocsc()
    mass_mat = scipy.sparse.diags(massvec)
    eigs_sigma = eps

    failcount = 0
    while True:
        try:
            evals, evecs = scipy.sparse.linalg.eigsh(
                lapl_eigsh, k=k_eig, M=mass_mat, sigma=eigs_sigma
            )
            evals = np.clip(evals, a_min=0.0, a_max=float("inf"))
            break
        except RuntimeError as exc:
            if failcount > 2:
                print("Fail", failcount)
                # raise ValueError("failed to compute eigendecomp") from exc
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            lapl_eigsh = lapl_eigsh + scipy.sparse.identity(lapl.shape[0]) * (
                eps * 10**failcount
            )
    return evals, evecs, lapl, massvec


def compute_tot_area(pos: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    side_1 = pos[faces[1]] - pos[faces[0]]
    side_2 = pos[faces[2]] - pos[faces[0]]
    return side_1.cross(side_2).norm(p=2, dim=1).abs().sum() / 2


def get_shape_condition(vertices, faces, dim):
    evals, evecs, lapl, massvec = compute_eig_laplacian(vertices, faces, dim)
    surf_area = compute_tot_area(torch.from_numpy(vertices), torch.from_numpy(faces))
    # Evals change with scale, follow first steps of cShapeDNA to get them
    # on a straight line and remove scale dependency.
    evals = torch.from_numpy(evals)
    arange = torch.arange(0, dim).float()
    # arange = torch.arange(0, evals.shape[1]).float()
    shape_desc = evals * surf_area - 4 * torch.pi * arange
    # shape_desc = evals * surf_area[:, None] - 4 * torch.pi * arange
>>>>>>> Stashed changes
    return shape_desc