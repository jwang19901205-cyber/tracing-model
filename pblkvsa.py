import cupy as cp

def kv_ob70_cupy(zz, pbl, kvmin=0.2, sbl=50.0, rktop=1.0):
    """
    基于 CuPy 的 O'Brien (1970) 剖面实现
    zz: 输入高度场, 形状 (nz, nx, ny)
    pbl: 边界层高度, 可以是标量或形状 (nx, ny) 的阵列
    """
    # 确保输入是 cupy 数组
    zz = cp.asarray(zz)
    pbl = cp.asarray(pbl)
    
    # 1. 计算地表梯度边界条件
    # 假设 zz[0] 是最底层高度
    dkdzs = kvmin / zz[0]
    ksbl = sbl * dkdzs
    
    # 2. 预计算 O'Brien 三次多项式项 (针对全空间)
    # 计算 (pbl - z) 和 (pbl - sbl) 相关的比例因子
    # 注意：为了防止除以0，分母加一个极小值
    denom = cp.maximum(pbl - sbl, 1.e-6)
    
    term_ratio = ((pbl - zz) / denom)**2
    term_linear = (ksbl - rktop + (zz - sbl) * (dkdzs + (2.0 * (ksbl - rktop) / denom)))
    
    obk = rktop + term_ratio * term_linear
    
    # 3. 使用 cp.where 构建分段函数 (核心逻辑)
    
    # 情况 A: z < sbl (地表层线性增长)
    res_sbl = cp.maximum(zz * dkdzs, kvmin)
    
    # 情况 B: sbl <= z < pbl (O'Brien 剖面)
    res_pbl = cp.maximum(obk, kvmin)
    
    # 情况 C: z >= pbl (边界层以上)
    res_free = 0.05
    
    # 组合最终结果
    rkv = cp.where(zz < sbl, res_sbl, 
          cp.where(zz < pbl, res_pbl, res_free))
    
    return rkv

# --- 使用示例 ---
# 假设 zz 是从其他模块传入的 (nz, nx, ny) 形状数组
# rkv = kv_ob70_cupy(zz_input, pbl_input)

def kv_ob70_cupy(zz, pbl, kvmin=0.2, sbl=50.0, rktop=1.0):
    """
    基于 CuPy 的 O'Brien (1970) 剖面实现（支持时间维度）
    
    Parameters:
    -----------
    zz : array-like, shape (nt, nz, nx, ny)
        输入高度场
    pbl : array-like, shape (nt, nx, ny)
        边界层高度
    kvmin : float, optional
        最小扩散系数，默认 0.2
    sbl : float, optional
        地表层厚度，默认 50.0
    rktop : float, optional
        边界层顶的扩散系数，默认 1.0
        
    Returns:
    --------
    rkv : cupy.ndarray, shape (nt, nz, nx, ny)
        湍流扩散系数剖面
    """
    # 确保输入是 cupy 数组
    zz = cp.asarray(zz)      # (nt, nz, nx, ny)
    pbl = cp.asarray(pbl)    # (nt, nx, ny)

    # 扩展 pbl 到四维以便与 zz 广播: (nt, 1, nx, ny)
    pbl_exp = pbl[:, None, :, :]  # 插入 nz 维度

    # 最底层高度 (z at level 0): shape (nt, nx, ny)
    z0 = zz[:, 0, :, :]  # (nt, nx, ny)

    # 防止除零：确保 z0 > 0
    z0 = cp.maximum(z0, 1e-6)

    # 地表梯度 dkdzs = kvmin / z0，形状 (nt, nx, ny)
    dkdzs = kvmin / z0

    # ksbl = sbl * dkdzs，形状 (nt, nx, ny)
    ksbl = sbl * dkdzs

    # 扩展标量 sbl 到与 zz 兼容的形状 (1, 1, 1, 1)，但更安全地用广播
    # denom = pbl - sbl，形状 (nt, nx, ny)
    denom = cp.maximum(pbl - sbl, 1e-6)  # (nt, nx, ny)

    # 扩展 denom 到 (nt, 1, nx, ny) 以匹配 zz
    denom_exp = denom[:, None, :, :]

    # 计算 (pbl - zz) -> 广播: (nt, nz, nx, ny)
    pbl_minus_zz = pbl_exp - zz  # (nt, nz, nx, ny)

    # term_ratio = ((pbl - zz) / denom)^2
    term_ratio = (pbl_minus_zz / denom_exp) ** 2  # (nt, nz, nx, ny)

    # 扩展 dkdzs 和 ksbl 到四维
    dkdzs_exp = dkdzs[:, None, :, :]   # (nt, 1, nx, ny)
    ksbl_exp = ksbl[:, None, :, :]     # (nt, 1, nx, ny)

    # term_linear = ksbl - rktop + (zz - sbl) * (dkdzs + 2*(ksbl - rktop)/denom)
    # 注意：(ksbl - rktop) 是 (nt, nx, ny)，需扩展
    ksbl_minus_rktop = ksbl - rktop  # (nt, nx, ny)
    ratio_k = 2.0 * ksbl_minus_rktop / denom  # (nt, nx, ny)
    ratio_k_exp = ratio_k[:, None, :, :]       # (nt, 1, nx, ny)

    term_linear = (ksbl_exp - rktop +
                   (zz - sbl) * (dkdzs_exp + ratio_k_exp))  # (nt, nz, nx, ny)

    obk = rktop + term_ratio * term_linear  # (nt, nz, nx, ny)

    # 情况 A: zz < sbl → res_sbl = max(zz * dkdzs, kvmin)
    res_sbl = cp.maximum(zz * dkdzs_exp, kvmin)

    # 情况 B: sbl <= zz < pbl → res_pbl = max(obk, kvmin)
    res_pbl = cp.maximum(obk, kvmin)

    # 情况 C: zz >= pbl → constant
    res_free = cp.full_like(zz, 0.01)

    # 组合分段函数
    rkv = cp.where(zz < sbl, res_sbl,
          cp.where(zz < pbl_exp, res_pbl, res_free))

    return rkv


def kv_ob70_4d_cupy(zz, pbl, kvmin=0.2, sbl=50.0, rktop=1.0):
    """
    基于 CuPy 的 O'Brien (1970) 剖面实现（支持时间维度）
    
    Parameters:
    -----------
    zz : array-like, shape (nt, nz, nx, ny)
        输入高度场
    pbl : array-like, shape (nt, nx, ny)
        边界层高度
    kvmin : float, optional
        最小扩散系数，默认 0.2
    sbl : float, optional
        地表层厚度，默认 50.0
    rktop : float, optional
        边界层顶的扩散系数，默认 1.0
        
    Returns:
    --------
    rkv : cupy.ndarray, shape (nt, nz, nx, ny)
        湍流扩散系数剖面
    """
    # 确保输入是 cupy 数组
    zz = cp.asarray(zz)      # (nt, nz, nx, ny)
    pbl = cp.asarray(pbl)    # (nt, nx, ny)

    # 扩展 pbl 到四维以便与 zz 广播: (nt, 1, nx, ny)
    pbl_exp = pbl[:, None, :, :]  # 插入 nz 维度

    # 最底层高度 (z at level 0): shape (nt, nx, ny)
    z0 = zz[:, 0, :, :]  # (nt, nx, ny)

    # 防止除零：确保 z0 > 0
    z0 = cp.maximum(z0, 1e-6)

    # 地表梯度 dkdzs = kvmin / z0，形状 (nt, nx, ny)
    dkdzs = kvmin / z0

    # ksbl = sbl * dkdzs，形状 (nt, nx, ny)
    ksbl = sbl * dkdzs

    # 扩展标量 sbl 到与 zz 兼容的形状 (1, 1, 1, 1)，但更安全地用广播
    # denom = pbl - sbl，形状 (nt, nx, ny)
    denom = cp.maximum(pbl - sbl, 1e-6)  # (nt, nx, ny)

    # 扩展 denom 到 (nt, 1, nx, ny) 以匹配 zz
    denom_exp = denom[:, None, :, :]

    # 计算 (pbl - zz) -> 广播: (nt, nz, nx, ny)
    pbl_minus_zz = pbl_exp - zz  # (nt, nz, nx, ny)

    # term_ratio = ((pbl - zz) / denom)^2
    term_ratio = (pbl_minus_zz / denom_exp) ** 2  # (nt, nz, nx, ny)

    # 扩展 dkdzs 和 ksbl 到四维
    dkdzs_exp = dkdzs[:, None, :, :]   # (nt, 1, nx, ny)
    ksbl_exp = ksbl[:, None, :, :]     # (nt, 1, nx, ny)

    # term_linear = ksbl - rktop + (zz - sbl) * (dkdzs + 2*(ksbl - rktop)/denom)
    # 注意：(ksbl - rktop) 是 (nt, nx, ny)，需扩展
    ksbl_minus_rktop = ksbl - rktop  # (nt, nx, ny)
    ratio_k = 2.0 * ksbl_minus_rktop / denom  # (nt, nx, ny)
    ratio_k_exp = ratio_k[:, None, :, :]       # (nt, 1, nx, ny)

    term_linear = (ksbl_exp - rktop +
                   (zz - sbl) * (dkdzs_exp + ratio_k_exp))  # (nt, nz, nx, ny)

    obk = rktop + term_ratio * term_linear  # (nt, nz, nx, ny)

    # 情况 A: zz < sbl → res_sbl = max(zz * dkdzs, kvmin)
    res_sbl = cp.maximum(zz * dkdzs_exp, kvmin)

    # 情况 B: sbl <= zz < pbl → res_pbl = max(obk, kvmin)
    res_pbl = cp.maximum(obk, kvmin)

    # 情况 C: zz >= pbl → constant
    res_free = cp.full_like(zz, 0.01)

    # 组合分段函数
    rkv = cp.where(zz < sbl, res_sbl,
          cp.where(zz < pbl_exp, res_pbl, res_free))

    return rkv.get()


def update_density_single_step(density, z, dx, dy, dz, PBLH, u_star, z0, cfl=0.1):
    """
    输入当前 3D 浓度和物理参数，计算单步扩散后的浓度场。
    
    参数:
    density: (nx, ny, nz) 的 CuPy 数组 (或者 Numpy 数组，会自动转换)
    z:       (nz,) 每层中心的高度数组 (m)
    dx, dy, dz: 网格间距 (m)
    PBLH:    边界层高度 (m)
    u_star:  地表摩擦速度 (m/s)
    z0:      地表粗糙度 (m)
    """
    # 确保数据在 GPU 上
    phi = cp.asarray(density, dtype=cp.float32)
    z = cp.asarray(z, dtype=cp.float32)
    nx, ny, nz = phi.shape
    kappa = 0.41

    # 1. 计算随高度变化的扩散系数 K(z)
    # 中性边界层公式: k = kappa * u_star * (z + z0)
    # 在 PBLH 以上，湍流扩散通常迅速减小至背景值（这里设为一个较小常数）
    k_z = kappa * u_star * (z + z0)
    k_z = cp.where(z <= PBLH, k_z, 0.01) # PBLH 以上设为极低扩散

    # 2. 自动计算该步的最佳 dt (基于稳定性判据)
    # dt < 0.5 * dz^2 / k_max
    k_max = cp.max(k_z)
    dt = cfl * (min(min(dx), min(dy), min(dz))**2) / k_max

    # 3. 垂直扩散计算 (垂直混合)
    # 计算交界面处的扩散系数 (使用邻层平均值)
    k_inter = 0.5 * (k_z[1:] + k_z[:-1])
    
    # 计算垂直通量 (Flux at interfaces): (nx, ny, nz-1)
    # F = -K * (d_phi / dz)
    z_flux = -k_inter * (phi[:, :, 1:] - phi[:, :, :-1]) / dz
    
    # 更新浓度 (散度算子): d_phi/dt = - d_flux/dz
    new_phi = cp.copy(phi)
    # 内部层更新 (1 到 nz-2)
    new_phi[:, :, 1:-1] -= dt * (z_flux[:, :, 1:] - z_flux[:, :, :-1]) / dz
    
    # 边界处理 (地面和顶部通量为 0)
    new_phi[:, :, 0]  -= dt * (z_flux[:, :, 0] - 0) / dz
    new_phi[:, :, -1] -= dt * (0 - z_flux[:, :, -1]) / dz

    # 4. 水平扩散计算 (假设水平扩散系数为垂直平均值的 10%)
    k_h = cp.mean(k_z) * 0.1
    
    # X 方向梯度
    x_flux = -k_h * (phi[1:, :, :] - phi[:-1, :, :]) / dx
    new_phi[1:-1, :, :] -= dt * (x_flux[1:, :, :] - x_flux[:-1, :, :]) / dx
    
    # Y 方向梯度
    y_flux = -k_h * (phi[:, 1:, :] - phi[:, :-1, :]) / dy
    new_phi[:, 1:-1, :] -= dt * (y_flux[:, 1:, :] - y_flux[:, :-1, :]) / dy

    return new_phi, dt



def compute_vdiff_dflux_gpu_fixed(density, dz, kv):
    """
    修正后的垂直扩散趋势项计算。
    确保输入 (nz, nx, ny) 输出也是 (nz, nx, ny)。
    """
    nz, nx, ny = density.shape
    
    # 1. 计算相邻 cell 中心点之间的距离 (nz-1 个距离)
    # dz_c[i] 是 density[i] 和 density[i+1] 之间的距离
    dz_c = 0.5 * (dz[:-1, :, :] + dz[1:, :, :])
    
    # 2. 计算界面上的梯度 (nz-1 个界面)
    # grad[i] 对应 kv[1:nz] 之间的界面
    grad_phi_int = (density[1:, :, :] - density[:-1, :, :]) / dz_c
    
    # 3. 计算内部界面通量 (nz-1 个)
    # kv 索引说明: kv[0]是底界, kv[nz]是顶界, kv[1:-1]是中间的 nz-1 个界面
    kvinter=0.5*( kv[:-1, :, :]+ kv[1:, :, :])
    flux_int = kvinter[:, :, :] * grad_phi_int
    
    # 4. 拼接边界通量 (实现 Zero-flux)
    # 底部 flux[0]=0, 顶部 flux[nz]=0, 中间是计算出的 nz-1 个 flux
    zero_shape = (1, nx, ny)
    flux_full = cp.concatenate([
        cp.zeros(zero_shape), 
        flux_int, 
        cp.zeros(zero_shape)
    ], axis=0) # 最终形状 (nz+1, nx, ny)
    
    # 5. 计算通量散度 (恢复到 nz 层)
    # dflux[i] = (Flux_interface_top[i+1] - Flux_interface_bottom[i]) / dz[i]
    # flux_full[1:] 是顶界面, flux_full[:-1] 是底界面
    dflux = (flux_full[1:, :, :] - flux_full[:-1, :, :]) / dz
    
    return dflux,flux_full

# def compute_vdiff_dflux_SA(flux_full,cp_p,dz):

#     nz=cp_p.shape[1]
#     dfluxsa=cp.zeros(cp_p)
#     for i in range (1,nz):
#         dfluxsa[:,i,:,:]= (cp.where(flux_full[i+1, :, :]>0, flux_full[i+1, :, :]*cp_p[:,i+1,:,:],flux_full[i+1, :, :]*cp_p[:,i,:,:])- cp.where(flux_full[i, :, :]>0,flux_full[i, :, :]*cp_p[:,i,:,:],flux_full[i, :, :]*cp_p[:,i-1,:,:])) / dz
    
#     dfluxsa[:,0,:,:]=(cp.where(flux_full[i+1, :, :]>0, flux_full[i+1, :, :]*cp_p[:,i+1,:,:],flux_full[i+1, :, :]*cp_p[:,i,:,:])-0)/dz
#     dfluxsa[:,-1,:,:]=(cp.where(flux_full[-1, :, :]>0, flux_full[-1, :, :]*0,flux_full[-1, :, :]*cp_p[:,-1,:,:])-cp.where(flux_full[-2, :, :]>0,flux_full[-2, :, :]*cp_p[:,-1,:,:],flux_full[-2, :, :]*cp_p[:,-2,:,:])) / dz
#     return dfluxsa


def compute_vdiff_dflux_SA(flux_full, cp_p, dz):
    # flux_full: (nz+1, ny, nx)
    # cp_p:       (nt, nz, ny, nx)

    # 扩展 cp_p 在垂直方向前后各补一层，用于 upwind/downwind 选择
    # cp_p_ext: (nt, nz+2, ny, nx)
    cp_p_ext = cp.pad(cp_p, ((0, 0), (1, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)

    # flux_full shape: (nz+1, ny, nx) → 广播为 (1, nz+1, ny, nx) 以便和 cp_p_ext 对齐
    flux = flux_full[None, :, :, :]  # (1, nz+1, ny, nx)

    # 上界面通量对应的 cp_p 值（upwind）
    # 若 flux[k] > 0，取 cp_p_ext[:, k+1]；否则取 cp_p_ext[:, k]
    # 注意：flux 的索引 k 对应界面 k（在层 k-1 和 k 之间）
    # 我们要计算每个层 i 的上界面 (i+1) 和下界面 (i)
    
    # 上界面（顶部通量）对应 cp_p 选择
    cp_upper = cp.where(flux[:, 1:, :, :] > 0,
                        cp_p_ext[:, 2:, :, :],   # i+1 层（因为 cp_p_ext[0] 是 padding）
                        cp_p_ext[:, 1:-1, :, :]) # i 层

    # 下界面（底部通量）对应 cp_p 选择
    cp_lower = cp.where(flux[:, :-1, :, :] > 0,
                        cp_p_ext[:, 1:-1, :, :], # i 层
                        cp_p_ext[:, :-2, :, :])  # i-1 层

    # 计算通量项
    flux_upper = flux[:, 1:, :, :] * cp_upper   # shape: (nt, nz, ny, nx)
    flux_lower = flux[:, :-1, :, :] * cp_lower  # shape: (nt, nz, ny, nx)

    # 边界修正：底层下通量设为 0；顶层上通量中 cp_p 设为 0（已在 pad 中实现）
    # 因为 cp_p_ext 第 0 层和最后一层是 0，所以下界面 i=0 时用 cp_p_ext[:,0] = 0（正确）
    # 上界面 i=nz-1 时，cp_upper 会用 cp_p_ext[:, nz+1] = 0（也正确）

    dfluxsa = (flux_upper - flux_lower) / dz

    return dfluxsa

