import numpy as np


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """
    Convert a quaternion into a rotation matrix.

    Parameters:
    q : array-like, shape (4,)
        Quaternion represented as [w, x, y, z]

    Returns:
    R : ndarray, shape (3, 3)
        Corresponding rotation matrix.
    """
    w, x, y, z = q
    R = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                  [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                  [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]])
    return R


def rotation_matrix_to_quaternion(R) -> np.ndarray:
    """
    Convert a rotation matrix into a quaternion.

    Parameters:
    R : array-like, shape (3, 3)
        Rotation matrix.

    Returns:
    q : ndarray, shape (4,)
        Corresponding quaternion represented as [w, x, y, z]
    """
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    trace = m00 + m11 + m22

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2  # S=4*qw
        w = 0.25 * S
        x = (m21 - m12) / S
        y = (m02 - m20) / S
        z = (m10 - m01) / S

    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2  # S=4*qx
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S

    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2  # S=4*qy
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S

    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2  # S=4*qz
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S

    return np.array([w, x, y, z])


def normalize(v) -> np.ndarray:
    """
    Normalize a vector.

    Parameters:
    v : array-like
        Input vector.

    Returns:
    v_normalized : ndarray
        Normalized vector.
    """
    norm = np.linalg.norm(v)
    
    return np.linalg.norm(v) if norm == 0 else v / norm


def triad(v1, v2, b1, b2) -> np.ndarray:
    """
    Compute rotation matrix using the TRIAD method.

    Parameters:
    v1 : array-like, shape (3,)
        First reference vector in the inertial frame.
    v2 : array-like, shape (3,)
        Second reference vector in the inertial frame.
    b1 : array-like, shape (3,)
        First measured vector in the body frame.
    b2 : array-like, shape (3,)
        Second measured vector in the body frame.

    Returns:
    R : ndarray, shape (3, 3)
        Rotation matrix from body frame to inertial frame.
    """
    # Normalize input vectors
    v1 = normalize(v1)
    v2 = normalize(v2)
    b1 = normalize(b1)
    b2 = normalize(b2)

    # Construct orthonormal bases
    t1 = v1
    t2 = normalize(np.cross(v1, v2))
    t3 = np.cross(t1, t2)

    b1_orth = b1
    b2_orth = normalize(np.cross(b1, b2))
    b3_orth = np.cross(b1_orth, b2_orth)

    # Form rotation matrices
    T = np.column_stack((t1, t2, t3))
    B = np.column_stack((b1_orth, b2_orth, b3_orth))

    # Compute rotation matrix
    R = T @ B.T

    return R


def project_unit_to_pixel(b, f, cx, cy) -> np.ndarray:
    """
    Project a unit vector in camera coordinates to pixel coordinates.

    Parameters:
    b : array-like, shape (3,)
        Unit vector in camera coordinates.
    f : float
        Focal length of the camera.
    cx : float
        x-coordinate of the principal point.
    cy : float
        y-coordinate of the principal point.

    Returns:
    p : ndarray, shape (2,)
        Pixel coordinates [u, v].
    """
    x, y, z = b
    u = f * (x / z) + cx
    v = f * (y / z) + cy
    return np.array([u, v])

def project_pixel_to_unit(u, v, f, cx, cy) -> np.ndarray:
    """
    Project pixel coordinates to a unit vector in camera coordinates.

    Parameters:
    u : float
        x-coordinate in pixel space.
    v : float
        y-coordinate in pixel space.
    f : float
        Focal length of the camera.
    cx : float
        x-coordinate of the principal point.
    cy : float
        y-coordinate of the principal point.

    Returns:
    b : ndarray, shape (3,)
        Unit vector in camera coordinates.
    """
    x = (u - cx) / f
    y = (v - cy) / f
    z = 1.0
    b = np.array([x, y, z])
    return normalize(b)


def synthetic_catalog(n=25, seed=0):
    """
    Generate a synthetic star catalog with random unit vectors.

    Parameters:
    n : int
        Number of stars to generate.
    seed : int
        Random seed for reproducibility.

    Returns:
    catalog : ndarray, shape (n, 3)
        Array of unit vectors representing star directions.
    """
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    mags = rng.uniform(1.0, 5.0, size=n)

    return vecs, mags


def generate_frame(q_true, f=800.0, cx=640, cy=480, FOV_deg=30.0, noise_px=0.3, seed=1):
    R = quaternion_to_rotation_matrix(q_true)
    S, mags = synthetic_catalog(n=60, seed=seed)
    B = (R @ S.T).T  # wektory w układzie kamery
    # filtrujemy tylko te w polu widzenia (b_z>0 i kąt < FOV/2)
    half_fov = np.deg2rad(FOV_deg/2)
    keep = (B[:,2] > np.cos(half_fov))  # z przodu i w stożku
    B = B[keep]; mags = mags[keep]
    # rzutowanie + szum
    pts = []
    rng = np.random.default_rng(seed+1)
    for b in B:
        u,v = project_unit_to_pixel(b, f, cx, cy)
        u += rng.normal(0, noise_px); v += rng.normal(0, noise_px)
        pts.append([u,v])
    return np.array(pts), B, S[keep]


def angle_error_deg(q_est, q_ref):
    # delta quaternion
    w1,x1,y1,z1 = q_est/np.linalg.norm(q_est)
    w2,x2,y2,z2 = q_ref/np.linalg.norm(q_ref)
    dot = abs(w1*w2 + x1*x2 + y1*y2 + z1*z2)
    dot = min(1.0, max(-1.0, dot))
    return 2*np.degrees(np.arccos(dot))


if __name__ == "__main__":
    # Prawda: lekka rotacja
    q_true = normalize(np.array([0.98, 0.05, 0.10, 0.10]))  # [w,x,y,z]
    pts, B, S_kept = generate_frame(q_true, FOV_deg=30)

    # Weźmy dwie najjaśniejsze gwiazdy: tu po prostu indeksy 0 i 1 (w docelowej wersji sortuj wg magnitudo/SNR)
    b1, b2 = normalize(B[0]), normalize(B[1])
    s1, s2 = normalize(S_kept[0]), normalize(S_kept[1])

    R_est = triad(s1, s2, b1, b2)
    q_est = rotation_matrix_to_quaternion(R_est)
    err = angle_error_deg(q_est, q_true)
    print(f"Błąd kątowy (TRIAD, 2 gwiazdy): {err:.3f} deg")