import numpy as np
import matplotlib.pyplot as plt


def agnesi(X,Y,xc,yc,ax,ay,h0):
    rr = ((X - xc) / ax) ** 2 + ((Y - yc) / ay) ** 2
    h = h0 / (1.0 + rr)
    return h

def stationary_mountain_wave_3d_rotating_nonhydro(
    nx,
    ny,
    nz,
    lx,
    ly,
    lz,
    U,
    N,
    f,
    h0,
    ax,
    ay,
):
    """
    3D stationary, rotating, nonhydrostatic linear mountain-wave model.

    Mean flow:
        U in +x direction

    Coordinates:
        x, y, z

    Variables returned:
        w, u, v, phi, b

    Governing steady 3D rotating nonhydrostatic relation:
        m^2 = (k^2 + l^2) * (N^2 - U^2 k^2) / (U^2 k^2 - f^2)

    Lower boundary condition:
        w(x,y,0) = U * dh/dx
        => w_hat(k,l,0) = i k U h_hat

    Notes
    -----
    - This is a spectral stationary solver in x and y.
    - Vertical structure is built analytically with exp(i m z).
    - The branch choice for m matters physically.
    - Modes near U^2 k^2 = f^2 are singular/ill-conditioned in the
      idealized inviscid theory; this code masks them.
    """

    # -------------------------------------------------
    # Grids
    # -------------------------------------------------
    x = np.linspace(0.001, lx, nx, endpoint=False)
    y = np.linspace(0.001, ly, ny, endpoint=False)
    z = np.linspace(0.001, lz, nz)

    dx = lx / nx
    dy = ly / ny

    X, Y = np.meshgrid(x, y, indexing="xy")

    # -------------------------------------------------
    # Terrain h(x,y)
    # -------------------------------------------------
    xc = lx / 2
    yc = ly / 2

    h = agnesi(X,Y,xc,yc,ax,ay,h0)
    h = h - np.mean(h)
    hmax=np.amax(h)
    # -------------------------------------------------
    # Horizontal spectral grids
    # -------------------------------------------------
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    l = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    k_grid, l_grid = np.meshgrid(k, l, indexing="xy")

    kh2_grid = k_grid**2 + l_grid**2
    Kh_grid = np.sqrt(kh2_grid)

    h_hat = np.fft.fft2(h)

    # -------------------------------------------------
    # Lower boundary forcing (linearised at z=0)
    #   w_hat(k,l,z=0) = i k U h_hat
    # -------------------------------------------------
    w0_hat = 1j * k_grid * U * h_hat

    # -------------------------------------------------
    # Vertical wavenumber m
    #   m^2 = (k^2 + l^2) * (N^2 - U^2 k^2) / (U^2 k^2 - f^2)
    #
    #   will do something smarter to remove singualrites (i.e. Uk=f)  
    # -------------------------------------------------
    denom = U**2 * k_grid**2 - f**2 
    m2 = kh2_grid * (N**2 - U**2 * k_grid**2) / denom
    m = np.zeros_like(k_grid, dtype=complex)
    propagating = m2 > 0
    evanescent = m2 < 0

    # Propagating branch:
    # choose sign so phase tilts appropriately with upward energy propagation
    m[propagating] = -np.sign(k_grid[propagating]) * np.sqrt(m2[propagating])

    # Evanescent branch:
    # choose decaying solution exp(i m z) = exp(-alpha z)
    m[evanescent] = 1j * np.sqrt(-m2[evanescent])


    # -------------------------------------------------
    # Other initial conditions from lower boundary forcing
    #   b_hat(k,l,z=0) = ...,  
    #   phi_hat(k,l,z=0) = ..., 
    #   u_hat(k,l,z=0) = ...,
    #   v_hat(k,l,z=0) = ....
    # -------------------------------------------------
    b0_hat = 1j * ( N**2 / (U * k_grid) ) * w0_hat
    b0_hat[np.isnan(b0_hat)] = 0
    phi0_hat = ( N**2 - U**2 * k_grid**2 )/( m * U * k_grid ) * w0_hat
    phi0_hat[np.isnan(phi0_hat)] = 0
    u0_hat = -( U * k_grid**2 - 1j * l_grid * f )/ denom * phi0_hat
    v0_hat = -( U * k_grid * l_grid + 1j * k_grid * f )/ denom * phi0_hat


    # -------------------------------------------------
    # Vertical structure
    # -------------------------------------------------
    w_hat_z = np.zeros((ny, nx, nz), dtype=complex)
    u_hat_z = np.zeros((ny, nx, nz), dtype=complex)
    v_hat_z = np.zeros((ny, nx, nz), dtype=complex)
    phi_hat_z = np.zeros((ny, nx, nz), dtype=complex)
    b_hat_z = np.zeros((ny, nx, nz), dtype=complex)

    for iz, zz in enumerate(z):
        vertical_factor = np.exp(1j * m * zz)

        w_hat_z[:, :, iz] = w0_hat * vertical_factor
        u_hat_z[:, :, iz] = u0_hat * vertical_factor
        v_hat_z[:, :, iz] = v0_hat * vertical_factor
        phi_hat_z[:, :, iz] = phi0_hat * vertical_factor
        b_hat_z[:, :, iz] = b0_hat * vertical_factor

    # -------------------------------------------------
    # Transform back to physical space
    # -------------------------------------------------
    w = np.zeros((ny, nx, nz), dtype=float)
    u = np.zeros((ny, nx, nz), dtype=float)
    v = np.zeros((ny, nx, nz), dtype=float)
    phi = np.zeros((ny, nx, nz), dtype=float)
    b = np.zeros((ny, nx, nz), dtype=float)

    for iz in range(nz):
        w[:, :, iz] = np.fft.ifft2(w_hat_z[:, :, iz]).real
        u[:, :, iz] = np.fft.ifft2(u_hat_z[:, :, iz]).real
        v[:, :, iz] = np.fft.ifft2(v_hat_z[:, :, iz]).real
        phi[:, :, iz] = np.fft.ifft2(phi_hat_z[:, :, iz]).real
        b[:, :, iz] = np.fft.ifft2(b_hat_z[:, :, iz]).real


    return {
        "x": x,
        "y": y,
        "z": z,
        "X": X,
        "Y": Y,
        "h": h,
        "w": w,
        "u": u,
        "v": v,
        "phi": phi,
        "b": b,
        "m": m,
        "m2": m2,
        "params": {
            "U": U,
            "N": N,
            "f": f,
            "lx": lx,
            "ly": ly,
            "lz": lz,
            "h0": h0,
            "ax": ax,
            "ay": ay,
            "nx": nx,
            "ny": ny,
            "nz": nz,
        },
    }




def plot_phi_slices(result, z_level_index=None, y_slice_index=None):

    x = result["x"]
    y = result["y"]
    z = result["z"]
    X = result["X"]
    Y = result["Y"]
    h = result["h"]
    phi = result["phi"]
    lz = result["params"]["lz"]

    if z_level_index is None:
        z_level_index = len(z) // 3
    if y_slice_index is None:
        y_slice_index = len(y) // 2

    z_plot = z[z_level_index]
    y_plot = y[y_slice_index]

    phi_horiz = phi[:, :, z_level_index]
    phi_vert = phi[y_slice_index, :, :]
    h_slice = h[y_slice_index, :]

    XZ, ZZ = np.meshgrid(x, z, indexing="xy")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    levels_h = np.linspace(phi_horiz.min(), phi_horiz.max(), 25)
    cf1 = axes[0].contourf(X, Y, phi_horiz, levels=levels_h)
    axes[0].contour(X, Y, h, levels=8, colors="k", linewidths=0.6)
    axes[0].set_title(f"Horizontal slice of phi at z = {z_plot:.3f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    #fig.colorbar(cf1, ax=axes[0], label="phi")

    levels_v = np.linspace(phi_vert.min(), phi_vert.max(), 25)
    cf2 = axes[1].contourf(XZ, ZZ, phi_vert.T, levels=levels_v)
    axes[1].contour(XZ, ZZ, phi_vert.T, levels=12, colors="k", linewidths=0.5)
    axes[1].fill_between(x, h_slice, 0, color="0.3")
    axes[1].set_title(f"Vertical slice of phi at y = {y_plot:.3f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    axes[1].set_ylim(0, lz)
    #fig.colorbar(cf2, ax=axes[1], label="phi")

    plt.tight_layout()
    plt.show()


def plot_uvw_slices(result, z_level_index=None, y_slice_index=None):

    x = result["x"]
    y = result["y"]
    z = result["z"]
    X = result["X"]
    Y = result["Y"]
    h = result["h"]
    u = result["u"]
    v = result["v"]
    w = result["w"]
    lz = result["params"]["lz"]

    if z_level_index is None:
        z_level_index = len(z) // 2
    if y_slice_index is None:
        y_slice_index = len(y) // 2

    z_plot = z[z_level_index]
    y_plot = y[y_slice_index]

    u_horiz = u[:, :, z_level_index]
    u_vert = u[y_slice_index, :, :]
    v_horiz = v[:, :, z_level_index]
    v_vert = v[y_slice_index, :, :]
    w_horiz = w[:, :, z_level_index]
    w_vert = w[y_slice_index, :, :]

    h_slice = h[y_slice_index, :]
    XZ, ZZ = np.meshgrid(x, z, indexing="xy")

    fig, axes = plt.subplots(3, 2, figsize=(9, 12))

    plt.title(r'$h_{peak}$ = {hmax:.3f}')

    # u horizontal
    levels = np.linspace(u_horiz.min(), u_horiz.max(), 25)
    cf1 = axes[0, 0].contourf(X, Y, u_horiz, levels=levels)
    axes[0, 0].contour(X, Y, h, levels=8, colors="k", linewidths=0.6)
    axes[0, 0].set_title(f"Horizontal slice of u at z = {z_plot:.3f}")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    fig.colorbar(cf1, ax=axes[0, 0], label="u")

    # u vertical
    levels = np.linspace(u_vert.min(), u_vert.max(), 25)
    cf2 = axes[0, 1].contourf(XZ, ZZ, u_vert.T, levels=levels)
    axes[0, 1].contour(XZ, ZZ, u_vert.T, levels=12, colors="k", linewidths=0.5)
    axes[0, 1].fill_between(x, h_slice, 0, color="0.3")
    axes[0, 1].set_title(f"Vertical slice of u at y = {y_plot:.3f}")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("z")
    axes[0, 1].set_ylim(0, lz)
    fig.colorbar(cf2, ax=axes[0, 1], label="u")

    # v horizontal
    levels = np.linspace(v_horiz.min(), v_horiz.max(), 25)
    cf3 = axes[1, 0].contourf(X, Y, v_horiz, levels=levels)
    axes[1, 0].contour(X, Y, h, levels=8, colors="k", linewidths=0.6)
    axes[1, 0].set_title(f"Horizontal slice of v at z = {z_plot:.3f}")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")
    fig.colorbar(cf3, ax=axes[1, 0], label="v")

    # v vertical
    levels = np.linspace(v_vert.min(), v_vert.max(), 25)
    cf4 = axes[1, 1].contourf(XZ, ZZ, v_vert.T, levels=levels)
    axes[1, 1].contour(XZ, ZZ, v_vert.T, levels=12, colors="k", linewidths=0.5)
    axes[1, 1].fill_between(x, h_slice, 0, color="0.3")
    axes[1, 1].set_title(f"Vertical slice of v at y = {y_plot:.3f}")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("z")
    axes[1, 1].set_ylim(0, lz)
    fig.colorbar(cf4, ax=axes[1, 1], label="v")
    
    # w horizontal
    levels = np.linspace(w_horiz.min(), w_horiz.max(), 25)
    cf5 = axes[2, 0].contourf(X, Y, w_horiz, levels=levels)
    axes[2, 0].contour(X, Y, h, levels=8, colors="k", linewidths=0.6)
    axes[2, 0].set_title(f"Horizontal slice of w at z = {z_plot:.3f}")
    axes[2, 0].set_xlabel("x")
    axes[2, 0].set_ylabel("y")
    fig.colorbar(cf5, ax=axes[2, 0], label="v")

    # w vertical
    levels = np.linspace(w_vert.min(), w_vert.max(), 25)
    cf6 = axes[2, 1].contourf(XZ, ZZ, w_vert.T, levels=levels)
    axes[2, 1].contour(XZ, ZZ, v_vert.T, levels=12, colors="k", linewidths=0.5)
    axes[2, 1].fill_between(x, h_slice, 0, color="0.3")
    axes[2, 1].set_title(f"Vertical slice of w at y = {y_plot:.3f}")
    axes[2, 1].set_xlabel("x")
    axes[2, 1].set_ylabel("z")
    axes[2, 1].set_ylim(0, lz)
    fig.colorbar(cf6, ax=axes[2, 1], label="v")


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    result = stationary_mountain_wave_3d_rotating_nonhydro(
        nx=256,
        ny=256,
        nz=256,
        lx=200000.,
        ly=200000.,
        lz=20000.,
        U=10.0,
        N=0.01,
        f=0.0001,
        h0=3500,
        ax=15000,
        ay=15000,
    )

    plot_phi_slices(result)
    plot_uvw_slices(result)
    

# Vary non-dimensional parameter, Froude number U/Nh for stability tests
# Illustrate Trapped vs propagating (maybe with Amplitude vs height animation)
# Phase tilt evolution 
# Breaking region?
# 2D vs 3D
# Quantify turbulence (from momemtum flux bar{u'w'}? ricahrdon number? PSD?)

# Propagating = wave can exist aloft, energy goes upward, m^2 > 0
# Trapped = wave cannot reach aloft, energy stays near the surface, m^2<0
# Breaking = wave becomes too large and collapses, wave collapses nonlinear, m*eta ~ 1 ( w = U deta/dx)

# Fr >> 1	weak stratification / strong flow
# Fr~1 nonlinear transition
# Fr << 1 strong stratification / weak flow
