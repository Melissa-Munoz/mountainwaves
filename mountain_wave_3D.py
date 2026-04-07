import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def agnesi(X,Y,xc,yc,ax,ay,h0):
    rr = ((X - xc) / ax) ** 2 + ((Y - yc) / ay) ** 2
    h = h0 / (1.0 + rr)
    return h

def gaussian(X,Y,xc,yc,ax,ay,hm):
    return hm*np.exp(- ( ( (X - xc) / ax) ** 2 + ( (Y - yc) / ay) ** 2))



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

    x_hgrid, y_hgrid = np.meshgrid(x, y, indexing="ij")
    x_vgrid, z_vgrid = np.meshgrid(x, z, indexing="ij")
    y_vgrid, z_Vgrid = np.meshgrid(y, z, indexing="ij")

    # -------------------------------------------------
    # Terrain h(x,y)
    # -------------------------------------------------
    xc = lx / 2
    yc = ly / 2

    h = agnesi(x_hgrid,y_hgrid,xc,yc,ax,ay,h0)
    h = h - np.mean(h)
    #h = gaussian(x_hgrid,y_hgrid,xc,yc,ax,ay,h0)
    #h = h - np.mean(h)
    hmax=np.amax(h)
    

    # -------------------------------------------------
    # Stability
    #   Nh/U > 1 leads to instabilities
    # -------------------------------------------------
    print(N*hmax/U)    
    
    
    # Horizontal spectral grids
    # -------------------------------------------------
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    l = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    k_grid, l_grid = np.meshgrid(k, l, indexing="ij")
    print(k)

    kh2_grid = k_grid**2 + l_grid**2
    Kh_grid = np.sqrt(kh2_grid)

    h_hat = np.fft.fft2(h)

    # -------------------------------------------------
    # Lower boundary forcing (linearised at z=0)
    #   w_hat(k,l,z=0) = i k U h_hat
    # -------------------------------------------------
    w0_hat = 1j * k_grid * U * h_hat
    #print(np.shape(k_grid),np.shape(h_hat))

    # -------------------------------------------------
    # Vertical wavenumber m
    #   m^2 = (k^2 + l^2) * (N^2 - U^2 k^2) / (U^2 k^2 - f^2)
    #
    #   will do something smarter to remove singualrites (i.e. Uk=f)  
    # -------------------------------------------------
    eps = 1e-12
    denom = U**2 * k_grid**2 - f**2 
    m2 = np.zeros((nx, ny), dtype=float)
    #m2 = kh2_grid * (N**2 - U**2 * k_grid**2) / denom
    nonzero_k = np.abs(k_grid) > eps
    nonzero_kh = kh2_grid > eps
    away_from_inertial = np.abs(denom) > eps
    active = nonzero_k & away_from_inertial & nonzero_kh
    m2[active] = kh2_grid[active] * (N**2 - U**2 * k_grid[active]**2) / denom[active]
    m = np.zeros((nx, ny), dtype=complex)
    propagating = active & (m2 > 0)
    evanescent = active & (m2 < 0)
    print('propagating',len(m[propagating]))
    print('evanesscent',len(m[evanescent]))
    
    # Propagating branch:
    # choose sign so phase tilts appropriately with upward energy propagation
    m[propagating] = -np.sign(k_grid[propagating]) * np.sqrt(m2[propagating])

    # Evanescent branch:
    # choose decaying solution exp(i m z) = exp(-alpha z)
    m[evanescent] = 1j * np.sqrt(-m2[evanescent])

    #m = np.where( m2 >= 0, np.sqrt(m2 + 0j), 1j * np.sqrt(-m2 + 0j))
    
    # -------------------------------------------------
    # Other initial conditions from lower boundary forcing
    #   b_hat(k,l,z=0) = ...,  
    #   phi_hat(k,l,z=0) = ..., 
    #   u_hat(k,l,z=0) = ...,
    #   v_hat(k,l,z=0) = ....
    # -------------------------------------------------
    phi0_hat = np.zeros_like(w0_hat, dtype=complex)
    u0_hat = np.zeros_like(w0_hat, dtype=complex)
    v0_hat = np.zeros_like(w0_hat, dtype=complex)
    b0_hat = np.zeros_like(w0_hat, dtype=complex)

    nonzero_m = np.abs(m) > eps
    valid = active & nonzero_m
    
    b0_hat[valid] = 1j * ( N**2 / (U * k_grid[valid]) ) * w0_hat[valid]
    phi0_hat[valid] = ( N**2 - U**2 * k_grid[valid]**2 )/( m[valid] * U * k_grid[valid] ) * w0_hat[valid]
    u0_hat[valid] = -( U * k_grid[valid]**2 - 1j * l_grid[valid] * f )/ denom[valid] * phi0_hat[valid]
    v0_hat[valid] = -( U * k_grid[valid] * l_grid[valid] + 1j * k_grid[valid] * f )/ denom[valid] * phi0_hat[valid]


    # -------------------------------------------------
    # Vertical structure
    # -------------------------------------------------
    w_hat_z = np.zeros((nx, ny, nz), dtype=complex)
    u_hat_z = np.zeros((nx, ny, nz), dtype=complex)
    v_hat_z = np.zeros((nx, ny, nz), dtype=complex)
    phi_hat_z = np.zeros((nx, ny, nz), dtype=complex)
    b_hat_z = np.zeros((nx, ny, nz), dtype=complex)

    for kk in range(nz):
        vertical_factor = np.exp(1j * m * z[kk])
        w_hat_z[:, :, kk] = w0_hat * vertical_factor
        u_hat_z[:, :, kk] = u0_hat * vertical_factor
        v_hat_z[:, :, kk] = v0_hat * vertical_factor
        phi_hat_z[:, :, kk] = phi0_hat * vertical_factor
        b_hat_z[:, :, kk] = b0_hat * vertical_factor

    print(np.shape(vertical_factor))

    # -------------------------------------------------
    # Transform back to physical space
    # -------------------------------------------------
    w = np.zeros((nx, ny, nz), dtype=float)
    u = np.zeros((nx, ny, nz), dtype=float)
    v = np.zeros((nx, ny, nz), dtype=float)
    phi = np.zeros((nx, ny, nz), dtype=float)
    b = np.zeros((nx, ny, nz), dtype=float)

    for kk in range(nz):
        w[:, :, kk] = np.fft.ifft2(w_hat_z[:, :, kk]).real
        u[:, :, kk] = np.fft.ifft2(u_hat_z[:, :, kk]).real
        v[:, :, kk] = np.fft.ifft2(v_hat_z[:, :, kk]).real
        phi[:, :, kk] = np.fft.ifft2(phi_hat_z[:, :, kk]).real
        b[:, :, kk] = np.fft.ifft2(b_hat_z[:, :, kk]).real


    return {
        "x": x,
        "y": y,
        "z": z,
        "x_hgrid": x_hgrid,
        "y_hgrid": y_hgrid,
        "x_vgrid": x_vgrid,
        "z_vgrid": z_vgrid,
        "y_vgrid": y_vgrid,
        "z_Vgrid": z_Vgrid,
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
            "xc": xc,
            "yc": yc,
               
        },
    }




def plot_phi_slices_streamlines(result, z_level_index=None, y_slice_index=None):

    x = result["x"]
    y = result["y"]
    z = result["z"]
    x_hgrid = result["x_hgrid"]
    y_hgrid = result["y_hgrid"]
    x_vgrid = result["x_vgrid"]
    z_vgrid = result["z_vgrid"]
    y_vgrid = result["y_vgrid"]
    z_Vgrid = result["z_Vgrid"]
    h = result["h"]
    u = result["u"]
    v = result["v"]
    w = result["w"]
    phi = result["phi"]
    lz = result["params"]["lz"]
    xc = result["params"]["xc"]
    yc = result["params"]["yc"]
    U = result["params"]["U"]
    h0 = result["params"]["h0"]
    ax = result["params"]["ax"]

    #u=U+u
    if z_level_index is None:
        z_level_index = 8#len(z) // 20
    if y_slice_index is None:
        y_slice_index = len(y) // 2
    print(z[z_level_index],np.amax(h))

    z_plot = z[z_level_index]
    y_plot = y[y_slice_index]

    phi_horiz = phi[:, :, z_level_index]
    phi_vert = phi[:,y_slice_index, :]
    h_slice = h[:,y_slice_index]

    u_h = u[:,:, z_level_index]
    v_h = v[:,:, z_level_index]
    #speed_h = np.sqrt(u_h**2 + v_h**2)

    u_v = u[:,y_slice_index,:]
    v_v = v[:,y_slice_index,:]
    w_v = w[:,y_slice_index,:]
    #speed_v = np.sqrt(u_v**2 + v_v**2)

    # For quiver  
    hstride = 8
    vstride = 8
    xq_hgrid = x_hgrid[::hstride,::hstride]
    yq_hgrid = y_hgrid[::hstride,::hstride]
    xq_vgrid = x_vgrid[::vstride,::vstride]
    zq_vgrid = z_vgrid[::vstride,::vstride]
    yq_vgrid = y_vgrid[::vstride,::vstride]
    zq_Vgrid = z_Vgrid[::vstride,::vstride]
    uq_hgrid = u_h[::hstride,::hstride]
    vq_hgrid = v_h[::hstride,::hstride]
    uq_vgrid = u_v[::vstride,::vstride]
    vq_vgrid = v_v[::vstride,::vstride]
    wq_vgrid = w_v[::vstride,::vstride]



    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    if z_plot<h0:
        radius_z = ax*np.sqrt( h0/z_plot - 1. )*1.1
        levels_h = np.linspace(phi_horiz.min(), phi_horiz.max(), 25)
        cf1 = axes[0].contourf(x_hgrid, y_hgrid, phi_horiz, levels=levels_h)
        axes[0].contour(x_hgrid, y_hgrid, h, levels=8, colors="k", linewidths=0.6)
        circle = Circle((xc,yc),radius = radius_z,color='0.3',zorder=10)
        axes[0].streamplot(x, y,u_h, v_h,density=1.5,color="k",linewidth=1.0,arrowsize=1.2)
        axes[0].set_title(f"Horizontal slice of phi at z = {z_plot:.3f}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].add_patch(circle)
        fig.colorbar(cf1, ax=axes[0], label="phi")
        
    if z_plot>h0:
        radius_z = ax*np.sqrt( h0/z_plot - 1. )*1.1
        levels_h = np.linspace(phi_horiz.min(), phi_horiz.max(), 25)
        cf1 = axes[0].contourf(x_hgrid, y_hgrid, phi_horiz, levels=levels_h)
        axes[0].contour(x_hgrid, y_hgrid, h, levels=8, colors="k", linewidths=0.6)
        axes[0].streamplot(x, y,u_h, v_h,density=1.5,color="k",linewidth=1.0,arrowsize=1.2)
        axes[0].set_title(f"Horizontal slice of phi at z = {z_plot:.3f}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(cf1, ax=axes[0], label="phi")    
        

    levels_v = np.linspace(phi_vert.min(), phi_vert.max(), 25)
    cf2 = axes[1].contourf(x_vgrid, z_vgrid, phi_vert, levels=levels_v)
    #axes[1].contour(x_vgrid, z_vgrid, phi_vert, levels=12, colors="k", linewidths=0.5)
    axes[1].streamplot(x, z,u_v.T, w_v.T,density=1.5,color="k",linewidth=1.0,arrowsize=1.2)
    axes[1].fill_between(x, h_slice, 0, color="0.3")
    axes[1].set_title(f"Vertical slice of phi at y = {y_plot:.3f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    axes[1].set_ylim(0, lz)
    fig.colorbar(cf2, ax=axes[1], label="phi")


    levels_v = np.linspace(phi_vert.min(), phi_vert.max(), 25)
    cf3 = axes[2].contourf(y_vgrid, z_Vgrid, phi_vert, levels=levels_v)
    #axes[1].contour(x_vgrid, z_vgrid, phi_vert, levels=12, colors="k", linewidths=0.5)
    axes[2].streamplot(y, z,u_v.T, w_v.T,density=1.5,color="k",linewidth=1.0,arrowsize=1.2)
    axes[2].fill_between(x, h_slice, 0, color="0.3")
    axes[2].set_title(f"Vertical slice of phi at x = {y_plot:.3f}")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("z")
    axes[2].set_ylim(0, lz)
    fig.colorbar(cf3, ax=axes[2], label="phi")

    plt.tight_layout()
    plt.show()


def plot_phi_slices_vectors(result, z_level_index=None, y_slice_index=None):

    x = result["x"]
    y = result["y"]
    z = result["z"]
    x_hgrid = result["x_hgrid"]
    y_hgrid = result["y_hgrid"]
    x_vgrid = result["x_vgrid"]
    z_vgrid = result["z_vgrid"]
    y_vgrid = result["y_vgrid"]
    z_Vgrid = result["z_Vgrid"]
    h = result["h"]
    u = result["u"]
    v = result["v"]
    w = result["w"]
    phi = result["phi"]
    lz = result["params"]["lz"]
    xc = result["params"]["xc"]
    yc = result["params"]["yc"]
    U = result["params"]["U"]
    h0 = result["params"]["h0"]
    ax = result["params"]["ax"]

    #u=U+u
    if z_level_index is None:
        z_level_index = 8#len(z) // 20
    if y_slice_index is None:
        y_slice_index = len(y) // 2
    print(z[z_level_index],np.amax(h))

    z_plot = z[z_level_index]
    y_plot = y[y_slice_index]

    phi_horiz = phi[:, :, z_level_index]
    phi_vert = phi[:,y_slice_index, :]
    h_slice = h[:,y_slice_index]

    u_h = u[:,:, z_level_index]
    v_h = v[:,:, z_level_index]
    #speed_h = np.sqrt(u_h**2 + v_h**2)

    u_v = u[:,y_slice_index,:]
    v_v = v[:,y_slice_index,:]
    w_v = w[:,y_slice_index,:]
    #speed_v = np.sqrt(u_v**2 + v_v**2)

    # For quiver  
    hstride = 6
    vstride = 6
    xq_hgrid = x_hgrid[::hstride,::hstride]
    yq_hgrid = y_hgrid[::hstride,::hstride]
    xq_vgrid = x_vgrid[::vstride,::vstride]
    zq_vgrid = z_vgrid[::vstride,::vstride]
    yq_vgrid = y_vgrid[::vstride,::vstride]
    zq_Vgrid = z_Vgrid[::vstride,::vstride]
    uq_hgrid = u_h[::hstride,::hstride]
    vq_hgrid = v_h[::hstride,::hstride]
    uq_vgrid = u_v[::vstride,::vstride]
    vq_vgrid = v_v[::vstride,::vstride]
    wq_vgrid = w_v[::vstride,::vstride]

    
    #mask = h > z_plot
    #phi_horiz_masked = np.ma.masked_where( mask,phi_horiz)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    if z_plot<h0:
        radius_z = ax*np.sqrt( h0/z_plot - 1. )*1.1
        levels_h = np.linspace(phi_horiz.min(), phi_horiz.max(), 25)
        cf1 = axes[0].contourf(x_hgrid, y_hgrid, phi_horiz, levels=levels_h)
        axes[0].contour(x_hgrid, y_hgrid, h, levels=8, colors="k", linewidths=0.6)
        circle = Circle((xc,yc),radius = radius_z,color='0.3',zorder=10)
        axes[0].quiver(xq_hgrid,yq_hgrid, uq_hgrid, vq_hgrid, angles='xy',scale_units='xy',color='k',width=0.002)
        axes[0].set_title(f"Horizontal slice of phi at z = {z_plot:.3f}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].add_patch(circle)
        fig.colorbar(cf1, ax=axes[0], label="phi")
        
    if z_plot>h0:
        radius_z = ax*np.sqrt( h0/z_plot - 1. )*1.1
        levels_h = np.linspace(phi_horiz.min(), phi_horiz.max(), 25)
        cf1 = axes[0].contourf(x_hgrid, y_hgrid, phi_horiz, levels=levels_h)
        axes[0].contour(x_hgrid, y_hgrid, h, levels=8, colors="k", linewidths=0.6)
        axes[0].quiver(xq_hgrid,yq_hgrid, uq_hgrid, vq_hgrid, angles='xy',scale_units='xy',color='k',width=0.002)
        axes[0].set_title(f"Horizontal slice of phi at z = {z_plot:.3f}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(cf1, ax=axes[0], label="phi")    
        

    levels_v = np.linspace(phi_vert.min(), phi_vert.max(), 25)
    cf2 = axes[1].contourf(x_vgrid, z_vgrid, phi_vert, levels=levels_v)
    axes[1].contour(x_vgrid, z_vgrid, phi_vert, levels=12, colors="k", linewidths=0.5)
    axes[1].quiver(xq_vgrid,zq_vgrid, uq_vgrid, wq_vgrid, angles='xy',scale_units='xy',color='k',width=0.002)
    axes[1].fill_between(x, h_slice, 0, color="0.3")
    axes[1].set_title(f"Vertical slice of phi at y = {y_plot:.3f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    axes[1].set_ylim(0, lz)
    fig.colorbar(cf2, ax=axes[1], label="phi")

    levels_v = np.linspace(phi_vert.min(), phi_vert.max(), 25)
    cf3 = axes[2].contourf(y_vgrid, z_Vgrid, phi_vert, levels=levels_v)
    axes[2].contour(x_vgrid, z_vgrid, phi_vert, levels=12, colors="k", linewidths=0.5)
    axes[2].quiver(yq_vgrid,zq_Vgrid, vq_vgrid, wq_vgrid, angles='xy',scale_units='xy',color='k',width=0.002)
    axes[2].fill_between(x, h_slice, 0, color="0.3")
    axes[2].set_title(f"Vertical slice of phi at x = {y_plot:.3f}")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("z")
    axes[2].set_ylim(0, lz)
    fig.colorbar(cf3, ax=axes[2], label="phi")

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    result = stationary_mountain_wave_3d_rotating_nonhydro(
        nx=128,
        ny=128,
        nz=128,
        lx=100000.,
        ly=100000.,
        lz=10000.,
        U=10.0,
        N=0.01,
        f=0.000001,
        h0=1000,#3500,
        ax=10000, #1500
        ay=10000, #1500
    )

    plot_phi_slices_streamlines(result)
    plot_phi_slices_vectors(result)
    #plot_uvw_slices(result)
    
