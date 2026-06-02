import cupy as cp

# 1. 核心计算内核 (已移除中文注释)
ppm_full_3d_kernel = cp.ElementwiseKernel(
    'T q_m2, T q_m1, T q, T q_p1, T q_p2, T q_m1_aR, T vel_face, T dt, T d_space',
    'T a_plus, T a_minus_raw',
    '''
    // A. 4th-order interface interpolation
    T aR = (7.0/12.0)*(q + q_p1) - (1.0/12.0)*(q_m1 + q_p2);
    T aL = q_m1_aR;

    // B. Limiter (C-S 2008)
    T d2q   = q_p1 + q_m1 - 2.0 * q;
    T d2q_m = q + q_m2 - 2.0 * q_m1;
    T d2q_p = q_p2 + q - 2.0 * q_p1;

    bool is_smooth = ((d2q > 0 && d2q_m > 0 && d2q_p > 0) || (d2q < 0 && d2q_m < 0 && d2q_p < 0));
    bool is_extrema = (q_p1 - q) * (q - q_m1) <= 0;

    if (is_extrema && !is_smooth) {
        aL = q; aR = q;
    } else if (is_smooth) {
        T q_max = max(q, max(q_m1, q_p1));
        T q_min = min(q, min(q_m1, q_p1));
        aL = (aL > q_max) ? q_max : ((aL < q_min) ? q_min : aL);
        aR = (aR > q_max) ? q_max : ((aR < q_min) ? q_min : aR);
    }

    // C. Monotonicity
    if (!is_smooth) {
        T da = aR - aL;
        T a6 = 6.0 * (q - 0.5 * (aL + aR));
        if (da * a6 > (da * da)) aL = 3.0 * q - 2.0 * aR;
        else if (da * a6 < -(da * da)) aR = 3.0 * q - 2.0 * aL;
    }

    // D. Characteristic Tracing
    T sigma = abs(vel_face * dt / d_space);
    T da_f = aR - aL;
    T a6_f = 6.0 * (q - 0.5 * (aL + aR));
    
    a_plus = aR - 0.5 * sigma * (da_f - (1.0 - (2.0/3.0) * sigma) * a6_f);
    a_minus_raw = aL + 0.5 * sigma * (da_f + (1.0 - (2.0/3.0) * sigma) * a6_f);
    ''',
    'ppm_full_3d_kernel'
)

def get_ppm_flux_3d_optimized(q, axis, vel_face, dt, d_space):
    q_m2 = cp.roll(q, 2, axis=axis)
    q_m1 = cp.roll(q, 1, axis=axis)
    q_p1 = cp.roll(q, -1, axis=axis)
    q_p2 = cp.roll(q, -2, axis=axis)

    a_face = (7.0/12.0)*(q + q_p1) - (1.0/12.0)*(q_m1 + q_p2)
    a_m1_aR = cp.roll(a_face, 1, axis=axis)

    a_plus, a_minus_raw = ppm_full_3d_kernel(
        q_m2, q_m1, q, q_p1, q_p2, 
        a_m1_aR, vel_face, dt, d_space
    )
    a_minus = cp.roll(a_minus_raw, -1, axis=axis)
    return cp.where(vel_face > 0, a_plus, a_minus)

def finalize_ctu_flux(rho_face, u_f, F_trans, dt, d_trans, blend, axis_long, axis_trans):
    """
    rho_face: 基础PPM界面值
    u_f: 沿axis_long方向的速度
    F_trans: 垂直方向(axis_trans)的通量
    """
    # 计算当前格点的横向通量梯度
    div_curr = (F_trans - cp.roll(F_trans, 1, axis=axis_trans)) / d_trans
    
    # 计算上游格点(沿axis_long方向)的横向通量梯度
    F_up = cp.roll(F_trans, 1, axis=axis_long)    # u > 0 情况
    F_down = cp.roll(F_trans, -1, axis=axis_long) # u < 0 情况
    
    div_up = (F_up - cp.roll(F_up, 1, axis=axis_trans)) / d_trans
    div_down = (F_down - cp.roll(F_down, 1, axis=axis_trans)) / d_trans
    
    div_final = cp.where(u_f > 0, div_curr, div_down)
    
    return u_f * (rho_face - 0.5 * dt * div_final * blend)

def compute_ctu_fluxes_3d(density, u_cent, v_cent, dt, dx, dy):
    # 假设输入为 (nz, ny, nx) -> axis 0, 1, 2
    
    # 1. 速度场插值
    u_face = 0.5 * (u_cent + cp.roll(u_cent, -1, axis=2)) # x-face
    v_face = 0.5 * (v_cent + cp.roll(v_cent, -1, axis=1)) # y-face

    # 2. 基础 PPM 状态
    rho_x = get_ppm_flux_3d_optimized(density, 2, u_face, dt, dx)
    rho_y = get_ppm_flux_3d_optimized(density, 1, v_face, dt, dy)

    F_x_tmp = u_face * rho_x
    F_y_tmp = v_face * rho_y

    # 3. Blend 计算
    blend_x = cp.clip(cp.abs(u_face * dt / dx) / 0.1, 0.0, 1.0)
    blend_y = cp.clip(cp.abs(v_face * dt / dy) / 0.1, 0.0, 1.0)

    # 4. 横向修正
    # Final Fx: x轴界面(axis=2)，受y轴通量(axis=1)修正
    final_fx = finalize_ctu_flux(rho_x, u_face, F_y_tmp, dt, dy, blend_x, axis_long=2, axis_trans=1)
    
    # Final Fy: y轴界面(axis=1)，受x轴通量(axis=2)修正
    final_fy = finalize_ctu_flux(rho_y, v_face, F_x_tmp, dt, dx, blend_y, axis_long=1, axis_trans=2)

    return final_fx, final_fy


def solver_step( fx, fy, dx, dy):
    """
    完整的平流求解步
    """
    # 计算修正后的通量
    
    # 严格守恒型更新: rho^{n+1} = rho^n - dt * div(Flux)
    # dfx = (F_{i+1/2} - F_{i-1/2}) / dx
    dfx = (fx - cp.roll(fx, 1, axis=2)) / dx
    dfy = (fy - cp.roll(fy, 1, axis=1)) / dy
    
    #new_density = density - dt * (dfx + dfy)
    
    # 质量守恒检查（可选）
    # total_mass = cp.sum(new_density) * dx * dy
    
    return dfx,dfy
# def solver_step_sa( fx, fy, dx, dy,cp_p):
#     fx_sa=cp.where(fx > 0, fx*cp_p, fx*cp.roll(cp_p, -1, axis=-1))
#     fy_sa=cp.where(fy > 0, fy*cp_p, fy*cp.roll(cp_p, -1, axis=-2))

#     dfx_sa = (fx_sa - cp.roll(fx_sa, 1, axis=-1)) / dx
#     dfy_sa = (fy_sa - cp.roll(fy_sa, 1, axis=-2)) / dy

#     return dfx_sa+dfy_sa

def solver_step_sa(fx, fy, dx, dy, cp_p):
    """
    fx, fy: 通量阵 (nz, ny, nx)
    cp_p: 组分比例 (np, nz, ny, nx)
    """
    # 增加维度以便与 cp_p (4D) 进行广播计算
    fx_4d = fx[None, ...] 
    fy_4d = fy[None, ...]

    # x方向迎风通量: axis=-1 是 nx
    # fx > 0 取当前格点 i, fx < 0 取下一个格点 i+1
    fx_sa = cp.where(fx_4d > 0, fx_4d * cp_p, fx_4d * cp.roll(cp_p, -1, axis=-1))
    
    # y方向迎风通量: axis=-2 是 ny
    # fy > 0 取当前格点 j, fy < 0 取下一个格点 j+1
    fy_sa = cp.where(fy_4d > 0, fy_4d * cp_p, fy_4d * cp.roll(cp_p, -1, axis=-2))

    # 计算通量散度 (Flux Divergence)
    # df = (Flux_{i+1/2} - Flux_{i-1/2}) / dx
    dfx_sa = (fx_sa - cp.roll(fx_sa, 1, axis=-1)) / dx
    dfy_sa = (fy_sa - cp.roll(fy_sa, 1, axis=-2)) / dy

    return dfx_sa + dfy_sa

