import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# Terrain shapes
# =============================================================================

def agnesi(X, Y, xc, yc, ax, ay, h0):
    rr = ((X - xc) / ax) ** 2 
    return h0 / (1.0 + rr)

def gaussian(X, Y, xc, yc, ax, ay, h0):
    return h0 * np.exp(-(((X - xc) / ax) ** 2 + ((Y - yc) / ay) ** 2))


# =============================================================================
# Solver
# =============================================================================

def stationary_mountain_wave_3d_rotating_nonhydro(
    nx, ny, nz,
    lx, ly, lz,
    U, N, f,
    h0, ax, ay,
    terrain='agnesi',
):
    """
    3D stationary, rotating, non-hydrostatic linear mountain-wave solver.

    Mean flow : U in +x direction only (V = 0)
    Terrain   : Agnesi witch (default) or Gaussian, centred in domain

    Dispersion relation (non-hydrostatic, rotating, steady):
        m^2 = (k^2 + l^2) * (N^2 - U^2*k^2) / (U^2*k^2 - f^2)

    Radiation condition for propagating modes (m^2 > 0):
        sgn(m) = sgn(k)   [upward energy propagation, U > 0]

    Evanescent modes (m^2 < 0):
        m = +i*|m|   [decays upward]

    Polarisation relations (derived from linearised Boussinesq eqs):
        w0_hat  = i * k * U * h_hat                               [terrain BC]
        b0_hat  = i * N^2 / (U*k) * w0_hat                       [thermodynamic]
        phi0_hat= (N^2 - U^2*k^2) / (m * U * k) * w0_hat         [vert. momentum]
        u0_hat  = -(U*k^2 - i*f*l) / (U^2*k^2 - f^2) * phi0_hat [horiz. momentum]
        v0_hat  = -(U*k*l  + i*f*k) / (U^2*k^2 - f^2) * phi0_hat

    NOTE on Hermitian symmetry
        u0_hat and v0_hat are anti-Hermitian with the raw polarisation sign.
        The -i factor is absorbed here so Re[IFFT] gives the correct real field.

    Displacements (for streamline plots):
        eta0_hat = w0_hat / (i*U*k)   [vertical,  z + eta = isentropes]
        xi0_hat  = v0_hat / (i*U*k)   [lateral,   y + xi  = horiz. streamlines]
    """

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------
    dx = lx / nx
    dy = ly / ny

    x = np.linspace(0, lx, nx, endpoint=False)
    y = np.linspace(0, ly, ny, endpoint=False)
    z = np.linspace(lz / nz, lz, nz)          # avoid z=0 singularity

    X, Y = np.meshgrid(x, y, indexing='ij')   # (nx, ny)

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    xc, yc = lx / 2, ly / 2

    if terrain == 'agnesi':
        h = agnesi(X, Y, xc, yc, ax, ay, h0)
    else:
        h = gaussian(X, Y, xc, yc, ax, ay, h0)

    h -= np.mean(h)                            # remove DC component
    hmax = np.amax(h)

    print(f"Nh/U = {N * hmax / U:.3f}  (linear theory valid when << 1)")

    # ------------------------------------------------------------------
    # Spectral wavenumbers
    # ------------------------------------------------------------------
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)  # (nx,)
    l = 2 * np.pi * np.fft.fftfreq(ny, d=dy)  # (ny,)

    k_grid, l_grid = np.meshgrid(k, l, indexing='ij')   # (nx, ny)
    kh2 = k_grid**2 + l_grid**2

    h_hat = np.fft.fft2(h)                     # (nx, ny), not normalised

    # ------------------------------------------------------------------
    # Lower boundary condition   w_hat(z=0) = i*k*U*h_hat
    # ------------------------------------------------------------------
    w0_hat = 1j * k_grid * U * h_hat

    # ------------------------------------------------------------------
    # Vertical wavenumber m(k,l)
    #
    #   m^2 = kh2 * (N^2 - U^2*k^2) / (U^2*k^2 - f^2)
    #
    # Active modes: k != 0,  kh2 != 0,  U^2*k^2 != f^2
    # ------------------------------------------------------------------
    eps = 1e-12

    Uk2   = U**2 * k_grid**2
    denom = Uk2 - f**2

    active = (
        (np.abs(k_grid) > eps) &
        (kh2            > eps) &
        (np.abs(denom)  > eps)
    )

    m2 = np.zeros((nx, ny), dtype=float)
    m2[active] = kh2[active] * (N**2 - Uk2[active]) / denom[active]

    propagating = active & (m2 >  0)
    evanescent  = active & (m2 <= 0) & active

    m = np.zeros((nx, ny), dtype=complex)

    # Radiation condition: sgn(m) = sgn(k)  for upward energy flux
    m[propagating] = np.sign(k_grid[propagating]) * np.sqrt(m2[propagating])

    # Evanescent: decays upward
    m[evanescent]  = 1j * np.sqrt(-m2[evanescent])

    print(f"Propagating modes : {propagating.sum()}")
    print(f"Evanescent  modes : {evanescent.sum()}")

    # ------------------------------------------------------------------
    # Polarisation relations at z = 0
    # ------------------------------------------------------------------
    valid = active & (np.abs(m) > eps)

    b0_hat   = np.zeros((nx, ny), dtype=complex)
    phi0_hat = np.zeros((nx, ny), dtype=complex)
    u0_hat   = np.zeros((nx, ny), dtype=complex)
    v0_hat   = np.zeros((nx, ny), dtype=complex)

    # Buoyancy  b = -N^2/(U*k) * w0_hat   (no i factor — must be Hermitian)
    b0_hat[valid] = -(N**2 / (U * k_grid[valid])) * w0_hat[valid]

    # Pressure  phi = (N^2 - U^2*k^2) / (m * U * k) * w0_hat
    phi0_hat[valid] = (
        (N**2 - Uk2[valid]) / (m[valid] * U * k_grid[valid])
    ) * w0_hat[valid]

    # Horizontal winds
    # Raw form has a leading -i making them anti-Hermitian → Re[IFFT] = 0.
    # Absorb the -i into the stored coefficient so the array is Hermitian.
    #   u_raw = -(Uk^2 - i*f*l) / denom * phi   ← anti-Hermitian
    #   u_stored = -i * u_raw                    ← Hermitian  ✓
    valid_uv = valid & (np.abs(denom) > eps)

    u0_hat[valid_uv] = np.divide(
        -(U * k_grid[valid_uv]**2 - 1j * f * l_grid[valid_uv]) * phi0_hat[valid_uv],
        denom[valid_uv]
    )
    v0_hat[valid_uv] = np.divide(
        -(U * k_grid[valid_uv] * l_grid[valid_uv] + 1j * f * k_grid[valid_uv]) * phi0_hat[valid_uv],
        denom[valid_uv]
    )

    # ------------------------------------------------------------------
    # Displacement spectra   (for streamline plots)
    #   eta_hat = w0_hat / (i*U*k)   [vertical displacement]
    #   xi_hat  = v0_hat / (i*U*k)   [lateral  displacement]
    # ------------------------------------------------------------------
    eta0_hat = np.zeros((nx, ny), dtype=complex)
    xi0_hat  = np.zeros((nx, ny), dtype=complex)

    eta0_hat[valid] = w0_hat[valid] / (1j * U * k_grid[valid])
    xi0_hat[valid]  = v0_hat[valid] / (1j * U * k_grid[valid])

    # ------------------------------------------------------------------
    # Vertical structure: multiply by e^{i*m*z} at each level
    # ------------------------------------------------------------------
    w   = np.zeros((nx, ny, nz))
    u   = np.zeros((nx, ny, nz))
    v   = np.zeros((nx, ny, nz))
    phi = np.zeros((nx, ny, nz))
    b   = np.zeros((nx, ny, nz))
    eta = np.zeros((nx, ny, nz))
    xi  = np.zeros((nx, ny, nz))

    for iz in range(nz):
        vf = np.exp(1j * m * z[iz])            # vertical phase factor (nx, ny)
        w  [:, :, iz] = np.fft.ifft2(w0_hat   * vf).real
        u  [:, :, iz] = np.fft.ifft2(u0_hat   * vf).real
        v  [:, :, iz] = np.fft.ifft2(v0_hat   * vf).real
        phi[:, :, iz] = np.fft.ifft2(phi0_hat * vf).real
        b  [:, :, iz] = np.fft.ifft2(b0_hat   * vf).real
        eta[:, :, iz] = np.fft.ifft2(eta0_hat * vf).real
        xi [:, :, iz] = np.fft.ifft2(xi0_hat  * vf).real

    print(f"max |w'|   = {np.max(np.abs(w)):.4f} m/s")
    print(f"max |eta'| = {np.max(np.abs(eta)):.1f} m")
    print(f"max |xi'|  = {np.max(np.abs(xi)):.1f} m")

    return {
        # grids
        "x": x, "y": y, "z": z,
        "X": X, "Y": Y,
        # terrain
        "h": h, "xc": xc, "yc": yc,
        # fields
        "w": w, "u": u, "v": v, "phi": phi, "b": b,
        # displacements
        "eta": eta, "xi": xi,
        # spectral
        "m": m, "m2": m2,
        "k": k, "l": l,
        # params
        "params": dict(
            nx=nx, ny=ny, nz=nz,
            lx=lx, ly=ly, lz=lz,
            U=U, N=N, f=f,
            h0=h0, ax=ax, ay=ay,
        ),
    }


def compute_nondimensional_numbers(result):

    p = result["params"]

    U  = p["U"]
    N  = p["N"]
    f  = p["f"]
    h0 = p["h0"]
    ax = p["ax"]

    # characteristic horizontal scale
    L = ax

    # characteristic wavenumber
    k = 2 * np.pi / L

    # Rossby
    Ro = U / (f * L) if f != 0 else np.inf

    # Froude
    Fr = U / (N * h0)
    
    # Scorer
    ls = N / U

    # Hydrostatic parameter
    H = N / (k * U)

    return Ro, Fr, ls, H

def plot_w_streamlines(result, iz_low=2, n_lines=18):

    import matplotlib.pyplot as plt
    import numpy as np

    x, y, z = result['x'], result['y'], result['z']
    w       = result['w']
    eta     = result['eta']
    xi      = result['xi']
    h       = result['h']

    iy   = len(y) // 2
    hmax = np.max(h)

    # --------------------------------------------------
    # Create subplots (horizontal LEFT, vertical RIGHT)
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))


    Ro, Fr, ls, H = compute_nondimensional_numbers(result)

    fig.suptitle(
        f"Numbers | Ro={Ro:.2f}, Fr={Fr:.2f}, "
        f"Scorer={ls:.3f}, Hydrostatic={H:.2f}",
        fontsize=13
    )
    # ==================================================
    # LEFT: Horizontal streamlines (x-y)
    # ==================================================
    ax = axes[0]

    zz_low = z[iz_low]
    w_xy   = w[:, :, iz_low]

    vmax = np.nanpercentile(np.abs(w_xy), 98) or 1

    cf = ax.contourf(
        x/1e3, y/1e3, w_xy.T,
        levels=40,
        cmap='RdBu_r',
        vmin=-vmax, vmax=vmax
    )

    # streamlines
    Y_stream = y[np.newaxis, :] + xi[:, :, iz_low]
    levels_y = np.linspace(y[0], y[-1], n_lines)

    ax.contour(
        x/1e3, y/1e3, Y_stream.T,
        levels=levels_y,
        colors='k',
        linewidths=1.2
    )

    # mountain footprint
    ax.contourf(
        x/1e3, y/1e3, h.T,
        levels=[hmax*0.1, hmax],
        colors=['0.7'],
        alpha=0.5
    )

    ax.contour(
        x/1e3, y/1e3, h.T,
        levels=[hmax*0.2, hmax*0.5, hmax*0.8],
        colors='k',
        linewidths=0.8
    )

    # limits
    ax.set_xlim(x[0]/1e3, x[-1]/1e3)
    ax.set_ylim(y[0]/1e3, y[-1]/1e3)

    ax.set_title(f"Horizontal streamlines at z={zz_low/1e3:.1f} km")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect('equal')

    # centered colorbar
    cbar = fig.colorbar(cf, ax=ax, shrink=1.0)
    cbar.ax.set_title("w (m/s)")

    # ==================================================
    # RIGHT: Vertical streamlines (x-z)
    # ==================================================
    ax = axes[1]

    w_xz = w[:, iy, :]

    vmax = np.nanpercentile(np.abs(w_xz), 98) or 1

    cf = ax.contourf(
        x/1e3, z/1e3, w_xz.T,
        levels=40,
        cmap='RdBu_r',
        vmin=-vmax, vmax=vmax
    )

    # streamlines
    Z_stream = z[np.newaxis, :] + eta[:, iy, :]
    levels_z = np.linspace(z[0], z[-1], n_lines)

    ax.contour(
        x/1e3, z/1e3, Z_stream.T,
        levels=levels_z,
        colors='k',
        linewidths=1.2
    )

    # mountain
    h_slice = h[:, iy]
    ax.fill_between(x/1e3, 0, h_slice/1e3, color='0.5')

    ax.set_xlim(x[0]/1e3, x[-1]/1e3)
    ax.set_ylim(0, z[-1]/1e3)

    ax.set_title(r"Vertical streamlines at $y=y_c$")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")

    # optional vertical exaggeration
    #ax.set_aspect(1.5)
    ax.set_aspect('auto')

    # centered colorbar
    cbar = fig.colorbar(cf, ax=ax, shrink=1.0)
    cbar.ax.set_title("w (m/s)")

    # --------------------------------------------------
    plt.tight_layout()
    plt.show()


def plot_phi_streamlines(result, iz_low=2, n_lines=18):

    import matplotlib.pyplot as plt
    import numpy as np

    x, y, z = result['x'], result['y'], result['z']
    phi     = result['phi']
    eta     = result['eta']
    xi      = result['xi']
    h       = result['h']

    iy   = len(y) // 2
    hmax = np.max(h)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ==================================================
    # LEFT: Horizontal slice (φ + horizontal streamlines)
    # ==================================================
    ax = axes[0]

    zz_low = z[iz_low]
    phi_xy = phi[:, :, iz_low]

    vmax = np.nanpercentile(np.abs(phi_xy), 98) or 1

    cf = ax.contourf(
        x/1e3, y/1e3, phi_xy.T,
        levels=40,
        cmap='RdBu_r',
        vmin=-vmax, vmax=vmax
    )

    # streamlines (y + ξ)
    Y_stream = y[np.newaxis, :] + xi[:, :, iz_low]
    levels_y = np.linspace(y[0], y[-1], n_lines)

    ax.contour(
        x/1e3, y/1e3, Y_stream.T,
        levels=levels_y,
        colors='k',
        linewidths=1.2
    )

    # mountain footprint
    ax.contourf(
        x/1e3, y/1e3, h.T,
        levels=[hmax*0.1, hmax],
        colors=['0.7'],
        alpha=0.5
    )

    ax.contour(
        x/1e3, y/1e3, h.T,
        levels=[hmax*0.2, hmax*0.5, hmax*0.8],
        colors='k',
        linewidths=0.8
    )

    ax.set_xlim(x[0]/1e3, x[-1]/1e3)
    ax.set_ylim(y[0]/1e3, y[-1]/1e3)

    ax.set_title(f"Horizontal streamlines at z={zz_low/1e3:.1f} km")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect('equal')

    cbar = fig.colorbar(cf, ax=ax, shrink=1.0)
    cbar.ax.set_title(r"$\phi (m^2/s^2)$")

    # ==================================================
    # RIGHT: Vertical slice (φ + vertical streamlines)
    # ==================================================
    ax = axes[1]

    phi_xz = phi[:, iy, :]

    vmax = np.nanpercentile(np.abs(phi_xz), 98) or 1

    cf = ax.contourf(
        x/1e3, z/1e3, phi_xz.T,
        levels=40,
        cmap='RdBu_r',
        vmin=-vmax, vmax=vmax
    )

    # streamlines (z + η)
    Z_stream = z[np.newaxis, :] + eta[:, iy, :]
    levels_z = np.linspace(z[0], z[-1], n_lines)

    ax.contour(
        x/1e3, z/1e3, Z_stream.T,
        levels=levels_z,
        colors='k',
        linewidths=1.2
    )

    # mountain
    h_slice = h[:, iy]
    ax.fill_between(x/1e3, 0, h_slice/1e3, color='0.5')

    ax.set_xlim(x[0]/1e3, x[-1]/1e3)
    ax.set_ylim(0, z[-1]/1e3)

    ax.set_title("Vertical streamlines at $y=y_c$")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_aspect('auto')  # important to avoid squishing

    cbar = fig.colorbar(cf, ax=ax, shrink=1.0)
    cbar.ax.set_title(r"$\phi (m^2/s^2)$")

    plt.tight_layout()
    plt.show()

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    result = stationary_mountain_wave_3d_rotating_nonhydro(
        nx=256,
        ny=256,
        nz=256,
        lx=200_000.,
        ly=200_000.,
        lz=10_000.,
        U=10.0,
        N=0.01,
        f=1e-4,
        h0=1000.,
        ax=10_000.,
        ay=10_000.,
        terrain='agnesi',
    )

    plot_w_streamlines(result, iz_low=2, n_lines=18)
    plot_phi_streamlines(result, iz_low=2, n_lines=18)
