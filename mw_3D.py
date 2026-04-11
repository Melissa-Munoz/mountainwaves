import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# Terrain shapes
# =============================================================================

def agnesi(X, Y, xc, yc, ax, ay, h0):
    rr = ((X - xc) / ax) ** 2 + ((Y - yc) / ay) ** 2
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
# Plot 3 — Hodograph: u'(z) vs v'(z) at selected (x,y) points
#
# The hodograph shape reveals the wave regime:
#   Line   → f = 0 (no rotation), u' and v' in phase
#   Ellipse → 0 < f < ω̂  (rotating IGW), axes ratio = f/ω̂
#   Circle  → f = ω̂  (inertial frequency, pure inertia-gravity wave)
#
# We plot hodographs at a few representative (x,y) locations —
# the mountain summit, windward flank, leeward flank, and a lateral point.
# Each curve is parametrised by z (coloured by height).
# =============================================================================
 
def plot_hodograph(result, n_points=4):
    x, y, z = result['x'], result['y'], result['z']
    u, v    = result['u'], result['v']
    h       = result['h']
    p       = result['params']
 
    nx, ny  = len(x), len(y)
    dx_     = x[1] - x[0]
    U_      = p['U']
    N_      = p['N']
    f_      = p['f']
    ax_     = p['ax']
 
    # Intrinsic frequency at dominant wavenumber k ~ 1/ax
    k_dom    = 2 * np.pi / ax_
    omega_dom = U_ * k_dom
    ratio     = f_ / omega_dom          # f/ω̂ → controls ellipse shape
 
    # ------------------------------------------------------------------
    # Choose representative (ix, iy) points
    # ------------------------------------------------------------------
    xc_idx = nx // 2                    # mountain centre x
    yc_idx = ny // 2                    # mountain centre y
    flank  = int(ax_ / dx_)            # one mountain half-width east
 
    points = [
        (xc_idx,          yc_idx,          'Summit'),
        (xc_idx + flank,  yc_idx,          'Leeward flank'),
        (xc_idx - flank,  yc_idx,          'Windward flank'),
        (xc_idx,          yc_idx + flank,  'Lateral offset'),
    ]
 
    # ------------------------------------------------------------------
    # Figure: 2×2 hodograph panels + 1 theory panel
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
 
    cmap_z = plt.cm.plasma                 # colour = height
 
    for idx, (ix, iy, label) in enumerate(points):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
 
        u_z = u[ix, iy, :]                # (nz,)  hodograph curve
        v_z = v[ix, iy, :]
 
        # Colour segments by height
        for iz in range(len(z) - 1):
            c = cmap_z(iz / len(z))
            ax.plot(u_z[iz:iz+2], v_z[iz:iz+2], color=c, linewidth=1.8)
 
        # Mark surface (z_min) and top (z_max)
        ax.plot(u_z[0],  v_z[0],  'o', color='lime',  ms=7,
                zorder=5, label=f'z={z[0]/1e3:.1f} km')
        ax.plot(u_z[-1], v_z[-1], 's', color='yellow', ms=7,
                zorder=5, label=f'z={z[-1]/1e3:.1f} km')
 
        # Theoretical ellipse axes from polarisation:
        #   minor axis / major axis = f / ω̂
        # Amplitude scales with local |u'|
        A = np.max(np.abs(u_z))           # major semi-axis ~ max u'
        B = ratio * A                     # minor semi-axis = (f/ω̂)*A
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(A * np.cos(theta), B * np.sin(theta),
                '--', color='#aaaaaa', linewidth=1.0, alpha=0.6,
                label=f'Theory  f/ω̂={ratio:.3f}')
 
        # Reference cross at origin
        ax.axhline(0, color='#444', linewidth=0.6)
        ax.axvline(0, color='#444', linewidth=0.6)
 
        ax.set_title(label, color=TC, fontsize=11, pad=6)
        ax.set_xlabel("u'  [m/s]", color=TC, fontsize=9)
        ax.set_ylabel("v'  [m/s]", color=TC, fontsize=9)
        ax.set_aspect('equal')
        ax.legend(fontsize=7, labelcolor=TC,
                  facecolor='#222', edgecolor='#444')
        _style(ax)
 
        # Colourbar for height
        sm = plt.cm.ScalarMappable(cmap=cmap_z,
             norm=plt.Normalize(vmin=z[0]/1e3, vmax=z[-1]/1e3))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('z [km]', color=TC, fontsize=8)
        cb.ax.yaxis.set_tick_params(color=TC)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TC)
 
    # ------------------------------------------------------------------
    # Fifth panel: regime diagram — f/ω̂ vs ellipse axis ratio
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[:, 2])      # spans both rows
 
    # Theoretical axis ratio = f/ω̂ for a range of k values
    k_range  = np.linspace(1e-5, 6 * k_dom, 500)
    om_range = U_ * k_range
    # only where N^2 > omega^2 > f^2  (propagating)
    prop_mask = (om_range**2 < N_**2) & (om_range**2 > f_**2)
    ratio_range = np.where(prop_mask, f_ / om_range, np.nan)
 
    ax5.plot(k_range[prop_mask] * ax_ / (2*np.pi),
             ratio_range[prop_mask],
             color='#4fc3f7', linewidth=2.0)
 
    # Mark dominant wavenumber
    ax5.axvline(1.0, color='yellow', linewidth=1.2, linestyle='--',
                label=f'k·ax/2π = 1  (dominant)')
    ax5.axhline(ratio, color='lime', linewidth=1.2, linestyle='--',
                label=f'Current f/ω̂ = {ratio:.4f}')
 
    # Shade regimes
    ax5.axhspan(0,    0.01,  alpha=0.08, color='grey',   label='Line-like  (f/ω̂ → 0)')
    ax5.axhspan(0.99, 1.01,  alpha=0.15, color='yellow', label='Circle  (f/ω̂ = 1)')
    ax5.axhspan(0,    1.0,   alpha=0.04, color='cyan')
 
    ax5.set_xlabel('k · ax / 2π  (normalised wavenumber)', color=TC, fontsize=10)
    ax5.set_ylabel('f / ω̂  =  ellipse minor/major axis ratio', color=TC, fontsize=10)
    ax5.set_title('Hodograph regime diagram', color=TC, fontsize=11, pad=8)
    ax5.set_ylim(0, 1.05)
    ax5.set_xlim(0, k_range[-1] * ax_ / (2*np.pi))
    ax5.legend(fontsize=8, labelcolor=TC, facecolor='#222', edgecolor='#444')
    _style(ax5)
 
    # Annotate regimes
    ax5.text(0.5, 0.03, 'Line-like',   color='#aaa', fontsize=9,
             transform=ax5.transAxes, ha='center')
    ax5.text(0.5, 0.55, 'Ellipse',     color='#4fc3f7', fontsize=10,
             transform=ax5.transAxes, ha='center')
    ax5.text(0.5, 0.96, 'Circle limit', color='yellow', fontsize=9,
             transform=ax5.transAxes, ha='center')
 
    fig.suptitle(
        f"Hodographs  u'(z) vs v'(z)  |  "
        f"U={U_} m/s  N={N_} rad/s  f={f_} rad/s  "
        f"f/ω̂(dominant) = {ratio:.4f}",
        color=TC, fontsize=12, fontweight='bold', y=0.995,
    )
    #plt.savefig('mountain_wave_hodograph.png', dpi=150,
    #            bbox_inches='tight', facecolor=BG)
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
