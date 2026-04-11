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


# =============================================================================
# Two-layer solver
# =============================================================================

def stationary_mountain_wave_3d_twolayer(
    nx, ny, nz,
    lx, ly, lz,
    U1, U2, z1,
    N, f,
    h0, ax, ay,
    terrain='agnesi',
    damp=0.05,
):
    """
    3D stationary, rotating, non-hydrostatic mountain-wave solver
    with a TWO-LAYER wind profile:

        U(z) = U1   for  0  <= z < z1    (lower layer)
        U(z) = U2   for  z1 <= z <= lz   (upper layer)

    N and f are constant throughout.

    Interface conditions at z = z1
    --------------------------------
    Two physical conditions must be satisfied:
        (1) Continuity of vertical displacement η = w / (i*ω̂)
        (2) Continuity of pressure φ

    Layer 1 carries an upward wave (A) and a downward reflected wave (B):
        w1(z) = A*exp(+i*m1*z) + B*exp(-i*m1*z)
        φ1(z) = Z1*[A*exp(+i*m1*z) - B*exp(-i*m1*z)]
                     (downward wave reverses impedance sign)

    Layer 2 carries only an upward wave (radiation condition):
        w2(z) = C*exp(i*m2*(z-z1))
        φ2(z) = Z2*C*exp(i*m2*(z-z1))

    where Z = (N² - ω̂²)/(ω̂*m) is the wave impedance.

    Three conditions for three unknowns (A, B, C):
        A + B = w0_hat                      [terrain BC at z=0]
        (A*e1 + B/e1)/ω̂1 = C/ω̂2           [η continuity at z=z1]
        Z1*(A*e1 - B/e1) = Z2*C             [φ continuity at z=z1]

    where e1 = exp(i*m1*z1).

    Solving gives:
        γ  = Z2*ω̂2 / (Z1*ω̂1)              (impedance ratio)
        R0 = (1 - γ) / (1 + γ)             (single-interface reflection)
        A  = w0_hat / (1 + R0*e1²)
        B  = R0 * e1² * A
        C  = (ω̂2/ω̂1) * (A*e1 + B/e1)

    Resonance / damping note
    ------------------------
    When |R0| = 1 (modes trapped in layer 1 because layer 2 is evanescent),
    the denominator 1 + R0*e1² can pass through zero — this is the linear
    resonance of a perfectly reflecting cavity and gives infinite amplitude
    in the absence of dissipation.

    We regularise with a small imaginary part added to m1:
        m1_eff = m1_real * (1 + i*damp)
    This gives |e1²| = exp(-2*damp*|m1|*z1) < 1, bounding the denominator.
    damp ~ 0.02–0.05 is standard (Lindzen & Tung 1976).
    """

    # ------------------------------------------------------------------
    # Grids
    # ------------------------------------------------------------------
    dx = lx / nx
    dy = ly / ny

    x = np.linspace(0, lx, nx, endpoint=False)
    y = np.linspace(0, ly, ny, endpoint=False)
    z = np.linspace(lz / nz, lz, nz)

    X, Y = np.meshgrid(x, y, indexing='ij')
    iz1  = np.searchsorted(z, z1)             # first index in layer 2

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    xc, yc = lx / 2, ly / 2

    if terrain == 'agnesi':
        h = agnesi(X, Y, xc, yc, ax, ay, h0)
    else:
        h = gaussian(X, Y, xc, yc, ax, ay, h0)

    h   -= np.mean(h)
    hmax = np.amax(h)
    print(f"Two-layer model  |  U1={U1} m/s  U2={U2} m/s  z1={z1/1e3:.1f} km")
    print(f"Nh/U1={N*hmax/U1:.3f}  Nh/U2={N*hmax/U2:.3f}  (linear valid when << 1)")

    # ------------------------------------------------------------------
    # Spectral wavenumbers
    # ------------------------------------------------------------------
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    l = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    k_grid, l_grid = np.meshgrid(k, l, indexing='ij')
    kh2 = k_grid**2 + l_grid**2

    h_hat = np.fft.fft2(h)
    w0_hat = 1j * k_grid * U1 * h_hat

    eps = 1e-12

    # ------------------------------------------------------------------
    # Per-layer helper: compute ω̂, m, Z, activity masks
    # damp_frac adds Im(m) = damp*Re(m) to propagating modes in layer 1
    # to regularise cavity resonance
    # ------------------------------------------------------------------
    def layer_setup(U_, damp_frac=0.0):
        omega  = U_ * k_grid
        omega2 = omega**2
        denom  = omega2 - f**2

        active = (
            (np.abs(k_grid) > eps) &
            (kh2            > eps) &
            (np.abs(denom)  > eps)
        )

        m2_arr = np.zeros((nx, ny))
        m2_arr[active] = kh2[active] * (N**2 - omega2[active]) / denom[active]

        prop = active & (m2_arr >  0)
        evan = active & (m2_arr <= 0) & active

        m = np.zeros((nx, ny), dtype=complex)
        m_real = np.sqrt(np.where(m2_arr > 0, m2_arr, 0.0))

        # Radiation condition + optional damping
        m[prop] = np.sign(k_grid[prop]) * m_real[prop] * (1.0 + 1j * damp_frac)
        m[evan] = 1j * np.sqrt(-m2_arr[evan])

        # Impedance Z = (N² - ω̂²) / (ω̂ * m)
        Z     = np.zeros((nx, ny), dtype=complex)
        valid = active & (np.abs(m) > eps)
        Z[valid] = (N**2 - omega2[valid]) / (omega[valid] * m[valid])

        return omega, m, Z, active, denom, prop, evan, valid

    om1, m1, Z1, act1, den1, prop1, evan1, valid1 = layer_setup(U1, damp_frac=damp)
    om2, m2, Z2, act2, den2, prop2, evan2, valid2 = layer_setup(U2, damp_frac=0.0)

    print(f"Layer 1  prop={prop1.sum()}  evan={evan1.sum()}")
    print(f"Layer 2  prop={prop2.sum()}  evan={evan2.sum()}")

    # ------------------------------------------------------------------
    # Interface matching
    # ------------------------------------------------------------------
    safe_Z = (np.abs(Z1) > eps) & (np.abs(om1) > eps)

    # Impedance ratio γ = Z2*ω̂2 / (Z1*ω̂1)
    gamma = np.zeros((nx, ny), dtype=complex)
    gamma[safe_Z] = Z2[safe_Z] * om2[safe_Z] / (Z1[safe_Z] * om1[safe_Z])

    # Single-interface reflection coefficient R0 = (1-γ)/(1+γ)
    R0 = (1 - gamma) / (1 + gamma + eps * 1j)   # small eps prevents 0/0

    # Phase factor at interface: e1 = exp(i*m1*z1)
    e1    = np.exp( 1j * m1 * z1)
    e1c   = np.exp(-1j * m1 * z1)
    e1sq  = e1 * e1                  # exp(2i*m1*z1), |e1sq| < 1 due to damping

    # Amplitude of upward wave in layer 1
    denom_A = 1.0 + R0 * e1sq
    safe_A  = act1 & (np.abs(denom_A) > eps)

    A = np.zeros((nx, ny), dtype=complex)
    A[safe_A] = w0_hat[safe_A] / denom_A[safe_A]

    # Reflected downward wave in layer 1
    B = R0 * e1sq * A

    # Transmitted upward wave in layer 2 (from η matching at z=z1)
    om1_safe = np.where(np.abs(om1) > eps, om1, 1.0)
    C = om2 / om1_safe * (A * e1 + B * e1c)

    print(f"max |R0|={np.abs(R0[act1]).max():.4f}  "
          f"max |A|={np.abs(A[act1]).max():.4f}  "
          f"max |C|={np.abs(C[act1&act2]).max():.4f}")

    # ------------------------------------------------------------------
    # Polarisation helper
    # ------------------------------------------------------------------
    def polar(w_hat, U_, m_layer, om_layer, den_layer, valid_mask, sign_m=1):
        """
        Compute phi, u, v, b, eta, xi spectral coefficients
        from w_hat using polarisation relations for given layer.
        sign_m = +1 for upward wave, -1 for downward (reflected).
        """
        Uk2_l  = U_**2 * k_grid**2
        v      = valid_mask & (np.abs(m_layer) > eps)
        vd     = v & (np.abs(den_layer) > eps)

        phi_h = np.zeros((nx, ny), dtype=complex)
        u_h   = np.zeros((nx, ny), dtype=complex)
        v_h   = np.zeros((nx, ny), dtype=complex)
        b_h   = np.zeros((nx, ny), dtype=complex)
        eta_h = np.zeros((nx, ny), dtype=complex)
        xi_h  = np.zeros((nx, ny), dtype=complex)

        # phi sign flips for downward wave (impedance reverses)
        phi_h[v] = sign_m * (N**2 - Uk2_l[v]) / (
            m_layer[v] * U_ * k_grid[v]) * w_hat[v]

        b_h[v]   = -(N**2 / (U_ * k_grid[v])) * w_hat[v]

        u_h[vd]  = -(U_ * k_grid[vd]**2 - 1j * f * l_grid[vd]) / den_layer[vd] * phi_h[vd]
        v_h[vd]  = -(U_ * k_grid[vd] * l_grid[vd] + 1j * f * k_grid[vd]) / den_layer[vd] * phi_h[vd]

        om_safe = np.where(np.abs(om_layer) > eps, om_layer, 1.0)
        eta_h[v] = w_hat[v] / (1j * om_safe[v])
        xi_h[v]  = v_h[v]  / (1j * om_safe[v])

        return phi_h, u_h, v_h, b_h, eta_h, xi_h

    # Layer 1: upward (A) and downward reflected (B)
    phi1A, u1A, v1A, b1A, eta1A, xi1A = polar(A, U1, m1, om1, den1, valid1, sign_m=+1)
    phi1B, u1B, v1B, b1B, eta1B, xi1B = polar(B, U1, m1, om1, den1, valid1, sign_m=-1)

    # Layer 2: upward only (C)
    phi2C, u2C, v2C, b2C, eta2C, xi2C = polar(C, U2, m2, om2, den2, valid2, sign_m=+1)

    # ------------------------------------------------------------------
    # Vertical structure
    # ------------------------------------------------------------------
    w   = np.zeros((nx, ny, nz))
    u   = np.zeros((nx, ny, nz))
    v   = np.zeros((nx, ny, nz))
    phi = np.zeros((nx, ny, nz))
    b   = np.zeros((nx, ny, nz))
    eta = np.zeros((nx, ny, nz))
    xi  = np.zeros((nx, ny, nz))

    for iz in range(nz):
        zz = z[iz]

        if zz < z1:
            # Layer 1: upward + downward
            vf_up = np.exp( 1j * m1 * zz)
            vf_dn = np.exp(-1j * m1 * zz)

            w_h   = A      * vf_up + B      * vf_dn
            phi_h = phi1A  * vf_up + phi1B  * vf_dn
            u_h   = u1A    * vf_up + u1B    * vf_dn
            v_h   = v1A    * vf_up + v1B    * vf_dn
            b_h   = b1A    * vf_up + b1B    * vf_dn
            eta_h = eta1A  * vf_up + eta1B  * vf_dn
            xi_h  = xi1A   * vf_up + xi1B   * vf_dn

        else:
            # Layer 2: upward only, phase from interface
            vf    = np.exp(1j * m2 * (zz - z1))

            w_h   = C      * vf
            phi_h = phi2C  * vf
            u_h   = u2C    * vf
            v_h   = v2C    * vf
            b_h   = b2C    * vf
            eta_h = eta2C  * vf
            xi_h  = xi2C   * vf

        w  [:, :, iz] = np.fft.ifft2(w_h  ).real
        u  [:, :, iz] = np.fft.ifft2(u_h  ).real
        v  [:, :, iz] = np.fft.ifft2(v_h  ).real
        phi[:, :, iz] = np.fft.ifft2(phi_h).real
        b  [:, :, iz] = np.fft.ifft2(b_h  ).real
        eta[:, :, iz] = np.fft.ifft2(eta_h).real
        xi [:, :, iz] = np.fft.ifft2(xi_h ).real

    print(f"max |w'|   = {np.max(np.abs(w)):.4f} m/s")
    print(f"max |eta'| = {np.max(np.abs(eta)):.1f} m")

    return {
        "x": x, "y": y, "z": z,
        "X": X, "Y": Y,
        "h": h, "xc": xc, "yc": yc,
        "w": w, "u": u, "v": v, "phi": phi, "b": b,
        "eta": eta, "xi": xi,
        "m1": m1, "m2": m2, "R0": R0,
        "k": k, "l": l,
        "z1": z1,
        "params": dict(
            nx=nx, ny=ny, nz=nz,
            lx=lx, ly=ly, lz=lz,
            U=U1, U1=U1, U2=U2, z1=z1,
            N=N, f=f, h0=h0, ax=ax, ay=ay,
        ),
    }



# =============================================================================
# Clean plotting helpers (white style)
# =============================================================================

def _style_clean(ax):
    ax.set_facecolor('white')
    ax.tick_params(colors='black', labelsize=9)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _cbar_clean(fig, ax, im, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)

def _contourf_clean(fig, ax, xi, yi, data, title, xlab, ylab, units):
    vmax = np.nanpercentile(np.abs(data), 98) or 1

    im = ax.contourf(
        xi / 1e3, yi / 1e3, data,
        levels=21,
        cmap='RdBu_r',
        vmin=-vmax, vmax=vmax
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    _style_clean(ax)
    _cbar_clean(fig, ax, im, units)

    return im
    

def plot_streamlines_clean(result, iz_low=2, n_lines=18):

    x, y, z = result['x'], result['y'], result['z']
    w       = result['w']
    eta, xi = result['eta'], result['xi']
    h       = result['h']

    iy   = len(y) // 2
    hmax = np.max(h)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    # --------------------------------------------------
    # LEFT: vertical streamlines
    # --------------------------------------------------
    ax = axes[0]

    w_xz = w[:, iy, :]

    _contourf_clean(fig, ax, x, z, w_xz.T,
                    "w'(x,z)", "x [km]", "z [km]", "m/s")

    # streamlines
    Z_stream = z[np.newaxis, :] + eta[:, iy, :]
    levels_z = np.linspace(z[0], z[-1], n_lines)

    ax.contour(x/1e3, z/1e3, Z_stream.T,
               levels=levels_z, colors='black', linewidths=1.0)

    # mountain
    ax.fill_between(x/1e3, 0, h[:, iy]/1e3, color='gray', alpha=0.6)

    ax.set_xlim(x[0]/1e3, x[-1]/1e3)
    ax.set_ylim(0, z[-1]/1e3)

    # --------------------------------------------------
    # RIGHT: horizontal streamlines
    # --------------------------------------------------
    ax = axes[1]

    w_xy = w[:, :, iz_low]

    _contourf_clean(fig, ax, x, y, w_xy.T,
                    f"w'(x,y) at z={z[iz_low]/1e3:.1f} km",
                    "x [km]", "y [km]", "m/s")

    Y_stream = y[np.newaxis, :] + xi[:, :, iz_low]
    levels_y = np.linspace(y[0], y[-1], n_lines)

    ax.contour(x/1e3, y/1e3, Y_stream.T,
               levels=levels_y, colors='black', linewidths=1.0)

    # mountain footprint
    ax.contourf(x/1e3, y/1e3, h.T,
                levels=[hmax*0.1, hmax*1.1],
                colors=['gray'], alpha=0.4)

    ax.set_xlim(x[0]/1e3, x[-1]/1e3)
    ax.set_ylim(y[0]/1e3, y[-1]/1e3)

    # --------------------------------------------------
    # Title
    # --------------------------------------------------
    p = result['params']
    fig.suptitle(
        f"Streamlines | U={p['U']}  N={p['N']}  f={p['f']}",
        fontsize=12
    )

    plt.tight_layout()
    plt.show()    


def plot_streamlines_comparison_clean(result_1L, result_2L, iz_low=2, n_lines=18):

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    for col, result in enumerate([result_1L, result_2L]):

        x, y, z = result['x'], result['y'], result['z']
        w       = result['w']
        eta     = result['eta']
        xi      = result['xi']
        h       = result['h']

        iy = len(y)//2
        hmax = np.max(h)

        # ------------------ vertical ------------------
        ax = axes[0, col]

        w_xz = w[:, iy, :]
        _contourf_clean(fig, ax, x, z, w_xz.T,
                        "Vertical", "x [km]", "z [km]", "m/s")

        Z_stream = z[np.newaxis, :] + eta[:, iy, :]
        ax.contour(x/1e3, z/1e3, Z_stream.T,
                   levels=np.linspace(z[0], z[-1], n_lines),
                   colors='black', linewidths=1)

        ax.fill_between(x/1e3, 0, h[:, iy]/1e3, color='gray', alpha=0.6)

        # ------------------ horizontal ------------------
        ax = axes[1, col]

        w_xy = w[:, :, iz_low]
        _contourf_clean(fig, ax, x, y, w_xy.T,
                        "Horizontal", "x [km]", "y [km]", "m/s")

        Y_stream = y[np.newaxis, :] + xi[:, :, iz_low]
        ax.contour(x/1e3, y/1e3, Y_stream.T,
                   levels=np.linspace(y[0], y[-1], n_lines),
                   colors='black', linewidths=1)

        ax.contourf(x/1e3, y/1e3, h.T,
                    levels=[hmax*0.1, hmax*1.1],
                    colors=['gray'], alpha=0.4)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    # --- Single-layer run ---
    result_1layer = stationary_mountain_wave_3d_rotating_nonhydro(
        nx=256, ny=256, nz=100,
        lx=200_000., ly=200_000., lz=10_000.,
        U=10.0, N=0.02, f=1e-4,
        h0=500., ax=10_000., ay=10_000.,
        terrain='agnesi',
    )

    #plot_overview(result_1layer)
    #plot_streamlines(result_1layer, iz_low=2)
    #plot_hodograph(result_1layer)

    # --- Two-layer run: slow lower layer, faster upper layer ---
    print("\n" + "="*55)
    result_2layer = stationary_mountain_wave_3d_twolayer(
        nx=256, ny=256, nz=100,
        lx=200_000., ly=200_000., lz=10_000.,
        U1=10.0,          # slow lower layer
        U2=60.0,         # faster upper layer
        z1=3_000.,       # interface height
        N=0.02, f=1e-4,
        h0=1000., ax=10_000., ay=10_000.,
        terrain='agnesi',
    )

    #plot_overview(result_2layer)
    plot_streamlines_clean(result_2layer, iz_low=2)

    # --- Side-by-side streamline comparison ---
    plot_streamlines_comparison_clean(result_1layer, result_2layer)
