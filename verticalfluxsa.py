import numpy as np
try:
    import cupy as cuy
except ImportError:
    cuy = None

def solve_implicit_vectorized(c_data, w_data, gph_data, dt):
# --- 正确的获取方法 ---
    if cuy is not None:
        xp = cuy.get_array_module(c_data)
    else:
        xp = np
    nz, ny, nx = c_data.shape

    # 计算 r = dt / dz
    r = dt / gph_data
    
    # 提取界面速度：w_up 是上界面 (k+0.5)，w_dn 是下界面 (k-0.5)
    w_up = w_data  # 假设 w_data[k] 是第 k 层的上界面
    w_dn = xp.zeros_like(w_up)
    w_dn[1:] = w_up[:-1]

    # --- 构造三对角矩阵系数 [nz, ny, nx] ---
    # b: 主对角线 (current level k)
    # a: 下对角线 (level k-1)
    # c: 上对角线 (level k+1)
    
    a = xp.zeros_like(c_data)
    b = xp.ones_like(c_data)
    c = xp.zeros_like(c_data)

    # 1. 填充主对角线 b (离开本层的量)
    # 向上流离开：w_up < 0
    b += r * xp.where(w_up < 0, -w_up, 0)
    # 向下流离开：w_dn > 0
    b += r * xp.where(w_dn > 0, w_dn, 0)

    # 2. 填充下对角线 a (从 k-1 进入 k 的量)
    # 向上流进入：w_dn < 0
    a[1:] = -r[1:] * xp.where(w_dn[1:] < 0, -w_dn[1:], 0)

    # 3. 填充上对角线 c (从 k+1 进入 k 的量)
    # 向下流进入：w_up > 0
    c[:-1] = -r[:-1] * xp.where(w_up[:-1] > 0, w_up[:-1], 0)

    # 4. 边界修正 (顶部和底部)
    # 已经在切片中通过 zeros_like 处理了边界不越界
    
    # 调用高效求解器
    c_new = thomas_solver_vec(a, b, c, c_data)
    
    return (c_new - c_data) / dt

def thomas_solver_vec(a, b, c, d):
    xp = cuy
    """
    向量化三对角求解器
    a, b, c, d 形状均为 [nz, ny, nx]
    """
    nz = a.shape[0]
    cp = xp.zeros_like(c)
    dp = xp.zeros_like(d)
    
    # Forward sweep (前向消元)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    
    for i in range(1, nz):
        denom = b[i] - a[i] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i-1]) / denom
        
    # Backward substitution (后向代换)
    x = xp.zeros_like(d)
    x[-1] = dp[-1]
    for i in range(nz-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]
        
    return x

def vertical_sa(dfux,cp_p):
    xp=cuy
    nz = dfux.shape[0]
    vdsa=xp.zeros_like(cp_p)
    dfluxtop=xp.zeros_like(dfux)
    dfluxbot=xp.zeros_like(dfux)
    vdsa[:,0,:,:]=dfux[0,:,:]*cp_p[:,0,:,:]
    dfluxtop[0,:,:]=-dfux[0,:,:]
    for i in range(1, nz-1):
        dfluxbot[i,:,:]=-dfluxtop[i-1,:,:]
        dfluxtop[i,:,:]=dfux[i,:,:]-dfluxbot[i,:,:]
        vdsa[:,i,:,:]= xp.where(dfluxbot[i,:,:]>0,dfluxbot[i,:,:]*cp_p[:,i-1,:,:],dfluxbot[i,:,:]*cp_p[:,i,:,:])+xp.where(dfluxtop[i,:,:]>0,dfluxtop[i,:,:]*cp_p[:,i+1,:,:],dfluxtop[i,:,:]*cp_p[:,i,:,:])
    vdsa[:,-1,:,:]=xp.where(dfluxbot[-1,:,:]>0,dfluxbot[-1,:,:]*cp_p[:,-2,:,:],dfluxbot[-1,:,:]*cp_p[:,-1,:,:])+xp.where(dfluxtop[-1,:,:]>0,dfluxtop[-1,:,:]*0,dfluxtop[-1,:,:]*cp_p[:,-1,:,:])
    return vdsa

def vertical_sa_optimized(dfux, cp_p):
    xp = cuy  # 假设 cp 是 numpy 或 cupy
    nz = dfux.shape[0]
    
    # 1. 预计算 dfluxtop 和 dfluxbot (向量化替代循环)
    # 通过前缀和 (cumsum) 逻辑可以一次性计算出所有通量边界
    # dfluxtop[i] = dfux[0] + dfux[1] + ... + dfux[i]
    dfluxtop = xp.cumsum(dfux, axis=0)
    
    # dfluxbot[i] 是上一个位置的 dfluxtop 的相反数
    # 我们构造一个错位数组
    dfluxbot = xp.zeros_like(dfux)
    dfluxbot[1:] = -dfluxtop[:-1]
    
    # 2. 初始化输出数组
    vdsa = xp.zeros_like(cp_p)
    
    # 3. 处理中间层 (i=1 到 nz-2) 的向量化计算
    # 使用切片代替循环
    # vdsa 维度通常为 (batch, depth, lat, lon)，所以 cp_p 索引对应 depth
    
    bot = dfluxbot[1:-1]
    top = dfluxtop[1:-1]
    
    # 上风格式逻辑处理
    term_bot = xp.where(bot > 0, bot * cp_p[:, :-2, :, :], bot * cp_p[:, 1:-1, :, :])
    term_top = xp.where(top > 0, top * cp_p[:, 2:, :, :],  top * cp_p[:, 1:-1, :, :])
    
    vdsa[:, 1:-1, :, :] = term_bot + term_top
    
    # 4. 处理边界层 (i=0 和 i=nz-1)
    # 第一层 (i=0)
    vdsa[:, 0, :, :] = dfux[0, :, :] * cp_p[:, 0, :, :]
    
    # 最后一层 (i=nz-1)
    bot_last = dfluxbot[-1]
    top_last = dfluxtop[-1]
    vdsa[:, -1, :, :] = xp.where(bot_last > 0, bot_last * cp_p[:, -2, :, :], bot_last * cp_p[:, -1, :, :]) + \
                        xp.where(top_last > 0, 0, top_last * cp_p[:, -1, :, :])
    
    return vdsa