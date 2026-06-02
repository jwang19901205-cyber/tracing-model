#
import numpy as np
import xarray as xr
from nmc_met_io.retrieve_cmadaas import cmadaas_model_grids
import warnings
from datetime import datetime, timedelta
import pandas as pd  
import os
import sys
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy is available. Using GPU acceleration.")
except ImportError:
    import numpy as cp
    CUPY_AVAILABLE = False
    print("CuPy not available. Falling back to CPU.")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy._lib.messagestream.MessageStream")

lat025 = np.arange(50, 34.8, -0.25)
lon025 = np.arange(100, 120.2, 0.25)
init_time = sys.argv[1]
#init_time = "2025051500"              # 起报时间: 
timenow = datetime.strptime(init_time, "%Y%m%d%H")
day_of_year0 = timenow.timetuple().tm_yday
endday=timenow + timedelta(days=2)
previous_day = timenow - timedelta(days=1)
previous_day_str = previous_day.strftime("%Y%m%d%H")
yfile_path = f"/mnt/g/FQDUST/dust_forecast_{previous_day_str}_phys-sigma-depbottom2-180s-wet-dt24-gtsi-met-bj.nc"


emisc=xr.open_dataset("/mnt/g/FQDUST/emiss"+init_time+"_gtsi_hourly2_bj.nc")
areac=xr.open_dataset("/mnt/g/SAND3D/grid_areas_dxdy.nc")
area=areac['area'].interp(lat=lat025, lon=lon025, method='nearest').values
dx=areac['dx'].interp(lat=lat025, lon=lon025, method='nearest').values*1000
dy=areac['dy'].interp(lat=lat025, lon=lon025, method='nearest').values*1000

emisssumf=xr.open_dataset('/mnt/g/dust/pm10_emiss_2025spring_sum.nc')
emisssum=emisssumf.pm10e_sum

dssand = xr.open_dataset('/mnt/g/dust/sand_usgs_0p25.nc')
sandd=dssand.sand
sandd=sandd.interp(lat=lat025, lon=lon025, method='nearest')
sandd= xr.where(sandd < 0, np.nan,sandd)
sandd = sandd.rename({
    'lon': 'longitude',
    'lat': 'latitude'
})

dsgsize=xr.open_dataset("/mnt/g/dust/gsize-23gtsi.nc")
gsize=dsgsize.gsize
gsizex=gsize.interp(lat=lat025, lon=lon025, method='nearest')

gsize=gsizex.values
# 多条件映射
conditions = [gsize <= 120, gsize == 150, gsize >= 180]
choices = [1.0, 1.0, 1.0]
ccr = np.select(conditions, choices, default=0.0)

dslaiy=xr.open_dataset("/mnt/g/dust/1000fvc/1000fvc/fvc_2024.nc")
laimin=dslaiy.Band1
laimin = laimin.interp(lat=lat025, lon=lon025, method='nearest')


dsluc = xr.open_dataset('/mnt/g/landuse/lanuse_modify_sand_0p05_228_61.nc')
gobi=dsluc.luc
gobi_ds025 = gobi.interp(lat=lat025, lon=lon025, method='nearest')
soiltype=xr.where(((gobi_ds025 == 61)& (laimin<0.07)),1.0,0.0)
#soiltype=xr.where(((laimin<0.1)),1.0,0.0)
soiltype = xr.where(gsizex==120,1.0,soiltype)
soiltype = soiltype.rename({
    'lon': 'longitude',
    'lat': 'latitude'
})

sandinitiallow=xr.where((soiltype==1) & (sandd >96),(103-(sandd+2))/(103-sandd),1.0)
sandinitialhigh = xr.where((soiltype==1) & (sandd >96),(103-(sandd-30))/(103-sandd),1.0)
print(day_of_year0)
if((day_of_year0>59)& (day_of_year0<166)):
    sandinitial= xr.where((soiltype==1) & (sandd >96),(103-(sandd-(30-0.1*(day_of_year0-60))))/(103-sandd),1.0)
elif(day_of_year0<60):
    sandinitial=sandinitialhigh
else:
    sandinitial=sandinitiallow
#sandl=xr.where((soiltype==1) & sandd >90,sandd-20,sandd)
#sandh=xr.where((soiltype==1) & sandd >90,sandd+2,sandd)
label_ds = xr.open_dataset('/mnt/g/dust/sand_region_labels_4.nc')
region_labels_np = label_ds['region_label'].values  # (ny, nx)
lat = label_ds['lat'].values
lon = label_ds['lon'].values
unique_labels = np.unique(region_labels_np)
n_labels = len(unique_labels)
region_labels_cp = cp.asarray(region_labels_np, dtype=cp.int32)


if os.path.exists(yfile_path):
    yccs=xr.open_dataset(yfile_path)
    target_time = timenow
    ccsy=yccs.ccs.sel(time=target_time).values
    fcsy=yccs.fcs.sel(time=target_time).values
    print('start ccs use previous forecast')
    if 'sandor' in yccs:
        sandor = yccs.sandor.sel(time=target_time)
    else:
        sandor = sandinitial
else:
    ccsy=np.zeros((13,61, 81))
    fcsy=np.zeros((13,61, 81))
    ccsay=np.zeros((n_labels,13,61, 81))
    print('start ccs use 0 ini')
    sandor = sandinitial


met=xr.open_dataset("/mnt/g/FQDUST/met_"+init_time+"_sigma_sv_bj.nc")

n_time = met.sizes['time']
n_lat = met.sizes['latitude']
n_lon = met.sizes['longitude']

#met2=xr.open_dataset("/mnt/g/SAND3D/met_rain_20250410.nc")
lsp=met['rain']
ustar=met['ustar']
dt24=met['dt24']
t2m=met['t2m']
sw=met['sw']
def calculate_vertical_velocity(u, v, gph, dx, dy):

    use_cp = CUPY_AVAILABLE and isinstance(u, np.ndarray)
    xp = cp if use_cp else np

    u_data = xp.asarray(u[:, ::-1, ::-1, :], dtype=xp.float64)
    v_data = xp.asarray(v[:, ::-1, ::-1, :], dtype=xp.float64)
    gph_data = xp.asarray(gph[:, ::-1, ::-1, :], dtype=xp.float64)
    dx_data = xp.asarray(dx[::-1, :], dtype=xp.float64)
    dy_data = xp.asarray(dy[::-1, :], dtype=xp.float64)

    t, z, m, n = u_data.shape

    # 插值 u 到东西边界
    u_interp = xp.zeros((t, z, m, n + 1))
    u_interp[:, :, :, 1:-1] = (u_data[:, :, :, :-1] + u_data[:, :, :, 1:]) / 2
    u_interp[:, :, :, 0] = u_data[:, :, :, 0]
    u_interp[:, :, :, -1] = u_data[:, :, :, -1]

    dx_padded = xp.zeros((z, m, n+1), dtype=dx_data.dtype)
    dx_padded[:, :, :-1] = dx_data
    dx_padded[:, :, -1] = dx_data[:,  -1]  # 注意是 [:, :, -1]

    zu_interp = xp.zeros((t, z, m, n + 1))
    zu_interp[:, :, :, 1:-1] = (gph_data[:, :, :, :-1] + gph_data[:, :, :, 1:]) / 2
    zu_interp[:, :, :, 0] = gph_data[:, :, :, 0]
    zu_interp[:, :, :, -1] = gph_data[:, :, :, -1]

    # 插值 v 到南北边界
    v_interp = xp.zeros((t, z, m + 1, n))
    v_interp[:, :, 1:-1, :] = (v_data[:, :, :-1, :] + v_data[:, :, 1:, :]) / 2
    v_interp[:, :, 0, :] = v_data[:, :, 0, :]
    v_interp[:, :, -1, :] = v_data[:, :, -1, :]

    dy_padded = xp.zeros((z,m+1, n))
    dy_padded[:,:-1, :]=dy_data[:,:]
    dy_padded[:,-1, :]=dy_data[-1,:]

    zv_interp = xp.zeros((t, z, m + 1, n))
    zv_interp[:, :, 1:-1, :] = (gph_data[:, :, :-1, :] + gph_data[:, :, 1:, :]) / 2
    zv_interp[:, :, 0, :] = gph_data[:, :, 0, :]
    zv_interp[:, :, -1, :] = gph_data[:, :, -1, :]

    F_east = u_interp / dx_padded * zu_interp
    F_south = v_interp / dy_padded * zv_interp

    mass_change = xp.zeros_like(u_data)
    mass_change -= F_east[:, :, :, 1:]
    mass_change += F_east[:, :, :, :-1]
    mass_change -= F_south[:, :, 1:, :]
    mass_change += F_south[:, :, :-1, :]

    divergence = mass_change

    w = xp.zeros_like(u_data)
    w_bottom = xp.zeros((t, m, n))

    for k in range(gph_data.shape[1]):
        current_div = divergence[:, k, :, :]
        w_top = current_div + w_bottom
        w[:, k, :, :] = w_top
        w_bottom = w_top

    w_result = w[:, ::-1, :, :]
    return xp.asnumpy(w_result) if use_cp else w_result


start_time = pd.Timestamp(met.time[0].values)  # 确保转换为pandas Timestamp  
end_time = start_time + pd.Timedelta(days=5)#pd.Timestamp(met.time[-1].values)  
new_time = pd.date_range(start=start_time, end=end_time, freq='h')  
print(start_time) 
print(end_time) 

sigma = met.sigma.values
sigma_mid = (sigma[:-1] + sigma[1:]) / 2
dsigma=sigma[:-1]-sigma[1]


wiu_mid = met['wiu'].interp(sigma=sigma_mid, method='linear')

wiv_mid = met['wiv'].interp(sigma=sigma_mid, method='linear')
gph_diff = -met.gph.diff("sigma", label="lower")

gph_0=met.gph.interp(sigma=sigma_mid, method='linear')
#gph_0=met.gph.interp(sigma=sigma_mid, method='linear')
gph_bottom=met.gph.sel(sigma=1.0)
wiw=calculate_vertical_velocity(wiu_mid.values,wiv_mid.values,gph_diff.values,dx,dy)
print(wiw.shape)

hours = met['time'].dt.hour
pbl_dict = {
    0: 900.,
    3: 1000.,
    6: 1200.,
    9: 1500.,
    12: 1200.,
    15: 1000.,
    18: 900.,
    21: 900.
}

pbl_values = [pbl_dict[hour] for hour in hours.values]
pbl_data = np.array(pbl_values)[:, np.newaxis, np.newaxis]  # 升维
pbl_data = np.tile(pbl_data, (1, n_lat, n_lon))  

coords = {
    "time": met.time.data,  
    "sigma": wiv_mid.sigma.data,
    "latitude": met.latitude.data,
    "longitude": met.longitude.data}

coordspbl = {
    "time": met.time.data,  
    "latitude": met.latitude.data,
    "longitude": met.longitude.data}

wiw=xr.DataArray(data=wiw,dims=('time', 'sigma', 'latitude', 'longitude'),coords=coords)
pbl=xr.DataArray(data=pbl_data,dims=('time',  'latitude', 'longitude'),coords=coordspbl)


def calculate_horizontal_divergence(u, v, c, gph, dx, dy, dt,cp_p):
    use_cp = CUPY_AVAILABLE
    xp = cp if use_cp else np
    u=xr.where(u>dx/dt,dx/dt-0.5,u)
    v=xr.where(v>dy/dt,dy/dt-0.5,v)
    u_data=u[:,::-1,:]
    v_data=v[:,::-1,:]
    c_data=c[:,::-1,:]
    cp_p_data=cp_p[:,:,::-1,:]
    gph_data=gph[:,::-1,:]
    dx_data=dx[::-1,:]
    dy_data=dy[::-1,:]   
    z,m, n = c_data.shape
    pad_width3 = ((0, 0), (1, 1), (1, 1))
    pad_width4 = ((0, 0),(0, 0), (1, 1), (1, 1))
    ca_padded = xp.pad(c_data, pad_width=pad_width3,mode='constant', constant_values=0.)
    cp_p_padded=xp.pad(cp_p_data, pad_width=pad_width4,mode='constant', constant_values=0.)
    #print('cp_p_data ',cp_p_padded.shape)
    # 将u插值到东/西边界 (m, n+1)
    u_interp = xp.zeros((z,m, n+1))
    u_interp[:,:, 1:-1] = (u_data[:,:, :-1] + u_data[:, :, 1:]) / 2  # 内部边界平均
    u_interp[:,:, 0] = u_data[:,:, 0]    # 西侧外边界
    u_interp[:,:, -1] = u_data[:,:, -1]  # 东侧外边界
    dx_padded = xp.zeros((z,m, n+1))
    dx_padded[:,:, :-1]=dx_data[:,:]
    dx_padded[:,:, -1]=dx_data[:,-1]
    zu_interp=xp.zeros((z,m, n+1))
    zu_interp[:,:, 1:-1] = (gph_data[:,:, :-1] + gph_data[:, :, 1:]) / 2  # 内部边界平均
    zu_interp[:,:, 0] = gph_data[:,:, 0]    # 西侧外边界
    zu_interp[:,:, -1] = gph_data[:,:, -1]  # 东侧外边界

    
    # 将v插值到南/北边界 (m+1, n)
    v_interp = xp.zeros((z,m+1, n))
    v_interp[:,1:-1, :] = (v_data[:,:-1, :] + v_data[:,1:, :]) / 2  # 内部边界平均
    v_interp[:,0, :] = v_data[:,0, :]    # 北侧外边界
    v_interp[:,-1, :] = v_data[:,-1, :]  # 南侧外边界
    dy_padded = xp.zeros((z,m+1, n))
    dy_padded[:,:-1, :]=dy_data[:,:]
    dy_padded[:,-1, :]=dy_data[-1,:]
    zv_interp=xp.zeros((z,m+1, n))
    zv_interp[:,1:-1, :] = (gph_data[:,:-1, :] + gph_data[:, 1:, :]) / 2  # 内部边界平均
    zv_interp[:, 0,:] = gph_data[:, 0,:]    # 西侧外边界
    zv_interp[:,-1,:] = gph_data[:,-1,:]  # 东侧外边界

    pad_width=1

    left_east = ca_padded[:,pad_width:-pad_width, pad_width-1 : -pad_width]
    left_east_sa=cp_p_padded[:,:,pad_width:-pad_width, pad_width-1 : -pad_width]
    right_east = ca_padded[:,pad_width:-pad_width, pad_width : None]
    right_east_sa=cp_p_padded[:,:,pad_width:-pad_width, pad_width : None]
    upstream_east = xp.where(u_interp > 0, left_east, right_east)
    upstream_east_sa =xp.where(u_interp > 0, left_east*left_east_sa, right_east*right_east_sa)
    F_east = u_interp * upstream_east/dx_padded*zu_interp
    F_east_sa = u_interp * upstream_east_sa/dx_padded*zu_interp
    
    # 计算南边界的上游浓度
    # 北侧浓度 (上方网格) 和南侧浓度 (下方网格)
    bot_south = ca_padded[:,pad_width-1 : -pad_width, pad_width:-pad_width]
    bot_south_sa=cp_p_padded[:,:,pad_width-1 : -pad_width, pad_width:-pad_width]
    top_south = ca_padded[:,pad_width : None, pad_width:-pad_width]
    top_south_sa =cp_p_padded[:,:,pad_width : None, pad_width:-pad_width]
    upstream_south = xp.where(v_interp > 0, bot_south, top_south)
    upstream_south_sa=xp.where(v_interp > 0, bot_south*bot_south_sa, top_south*top_south_sa)
    F_south = v_interp *  upstream_south/dy_padded*zv_interp
    F_south_sa=v_interp *  upstream_south_sa/dy_padded*zv_interp
    
    # 计算质量变化
    mass_change = xp.zeros_like(c_data)
    mass_change_sa=xp.zeros_like(cp_p_data)
    # 东-西方向：左侧网格流出，右侧网格流入
    mass_change[:,:, :] -= F_east[:,:, 1:]   # 左侧减少
    mass_change[:,:, :] += F_east[:,:, :-1]      # 右侧增加

    mass_change_sa[:,:,:,:] -=F_east_sa[:,:,:, 1:]
    mass_change_sa[:,:,:,:] +=F_east_sa[:,:,:, :-1]
    # 南-北方向：上方网格流出，下方网格流入
    mass_change[:,:, :] -= F_south[:,1:, :]     # 上方减少
    mass_change[:,:, :] += F_south[:,:-1, :]    # 下方增加
    mass_change_sa[:,:,:,:] -=F_south_sa[:,:,1:, :]
    mass_change_sa[:,:,:,:] +=F_south_sa[:,:,:-1, :]

    mass_change=mass_change/gph_data
    mass_change_sa=mass_change_sa/gph_data

    return mass_change[:, ::-1, :],mass_change_sa[:,:,::-1,:]


def calculate_vertical_divergence(w, c, gph,cp_p):
    use_cp = CUPY_AVAILABLE
    xp = cp if use_cp else np
    wdata=w[::-1,:,:]
    cdata=c[::-1,:,:]
    cp_pdata=cp_p[:,::-1,:,:]
    gphdata=gph[::-1,:,:]
    fw=xp.zeros_like(wdata)
    fwsa=xp.zeros_like(cp_pdata)
    fw_botoom=xp.zeros_like(wdata[ 0, :, :])
    fw_botoomsa=xp.zeros_like(cp_pdata[:, 0, :, :])
    for k in range(wdata.shape[0]-1):
        fw[k,:,:]=(xp.where(wdata[k,:,:]>0,-wdata[k,:,:]*cdata[k,:,:],-wdata[k,:,:]*cdata[k+1,:,:])+fw_botoom)/gphdata[k,:,:]
        fwsa[:,k,:,:]=(xp.where(wdata[k,:,:]>0,-wdata[k,:,:]*cdata[k,:,:]*cp_pdata[:,k,:,:],-wdata[k,:,:]*cdata[k+1,:,:])*cp_pdata[:,k+1,:,:]+fw_botoomsa)/gphdata[k,:,:]

        fw_botoom=-xp.where(wdata[k,:,:]>0,-wdata[k,:,:]*cdata[k,:,:],-wdata[k,:,:]*cdata[k+1,:,:])
        fw_botoomsa=-xp.where(wdata[k,:,:]>0,-wdata[k,:,:]*cdata[k,:,:]*cp_pdata[:,k,:,:],-wdata[k,:,:]*cdata[k+1,:,:]**cp_pdata[:,k+1,:,:])

    fw[k+1,:,:]=(xp.where(wdata[k+1,:,:]>0,-wdata[k+1,:,:]*cdata[k+1,:,:],-wdata[k+1,:,:]*0)+fw_botoom)/gphdata[k+1,:,:]
    fwsa[:,k+1,:,:]=(xp.where(wdata[k+1,:,:]>0,-wdata[k+1,:,:]*cdata[k+1,:,:]*cp_pdata[:,k+1,:,:],-wdata[k+1,:,:]*0)+fw_botoomsa)/gphdata[k+1,:,:]

    #dfw=xr.DataArray(data=xp.asnumpy(fw[::-1,:,:]),dims=('sigma', 'latitude', 'longitude'),coords=coords1)
    return fw[::-1,:,:],fwsa[:,::-1,:,:]


def calculate_vertical_diffusion(w, c, kv,gph,cp_p):
    use_cp = CUPY_AVAILABLE
    xp = cp if use_cp else np

    wdata=w[::-1,:,:]
    cdata=c[::-1,:,:]
    cp_pdata=cp_p[:,::-1,:,:]
    gphdata=gph[::-1,:,:]
    kvdata=kv[::-1,:,:]
    fwf=xp.zeros_like(wdata)
    fwfsa=xp.zeros_like(cp_pdata)
    fwf_botoom=xp.zeros_like(wdata[ 0, :, :])
    fwf_botoomsa=xp.zeros_like(cp_pdata[ :,0, :, :])

    for k in range(wdata.shape[0]-1):
        ftop=(cdata[k+1,:,:]-cdata[k,:,:])/(gphdata[k,:,:]+gphdata[k+1,:,:])*kvdata[k,:,:]*(gphdata[k,:,:]*gphdata[k+1,:,:])
        fwf[k,:,:]=((cdata[k+1,:,:]-cdata[k,:,:])/(gphdata[k,:,:]+gphdata[k+1,:,:])*kvdata[k,:,:]*(gphdata[k,:,:]*gphdata[k+1,:,:])-fwf_botoom)/gphdata[k,:,:]
        fwfsa[:,k,:,:]=xp.where(ftop>0,(ftop*cp_pdata[:,k+1,:,:]-fwf_botoomsa)/gphdata[k,:,:],(ftop*cp_pdata[:,k,:,:]-fwf_botoomsa)/gphdata[k,:,:])        
        fwf_botoom=(cdata[k+1,:,:]-cdata[k,:,:])/(gphdata[k,:,:]+gphdata[k+1,:,:])*kvdata[k,:,:]*(gphdata[k,:,:]*gphdata[k+1,:,:])
        fwf_botoomsa=xp.where(fwf_botoom>0,fwf_botoom*cp_pdata[:,k+1,:,:],fwf_botoom*cp_pdata[:,k,:,:])
    fwf[k+1,:,:]=((0.-cdata[k+1,:,:])/(gphdata[k+1,:,:]+gphdata[k+1,:,:])*kvdata[k+1,:,:]*(gphdata[k+1,:,:]*gphdata[k+1,:,:])-fwf_botoom)/gphdata[k+1,:,:]
    fwfsa[:,k+1,:,:]=((0.-cdata[k+1,:,:]*cp_pdata[:,k+1,:,:])/(gphdata[k+1,:,:]+gphdata[k+1,:,:])*kvdata[k+1,:,:]*(gphdata[k+1,:,:]*gphdata[k+1,:,:])-fwf_botoomsa)/gphdata[k+1,:,:]
    #dfwf=xr.DataArray(data=xp.asnumpy(fwf[::-1, :, :]) if use_cp else fwf[::-1, :, :],dims=('sigma', 'latitude', 'longitude'),coords=coords1)
    return fwf[::-1,:,:],fwfsa[:,::-1,:,:]

emisc=emisc.interp(time=new_time, method='linear')
wiu_mid=wiu_mid.sortby('time')
wiuh=wiu_mid.interp(time=new_time, method='linear')
wiuh = wiuh.interpolate_na(dim="longitude", method="linear")

wiv_mid=wiv_mid.sortby('time')
wivh=wiv_mid.interp(time=new_time, method='linear')
wivh = wivh.interpolate_na(dim="longitude", method="linear")

gph_diff=gph_diff.sortby('time')
gphh=gph_diff.interp(time=new_time, method='linear')
gphh = gphh.interpolate_na(dim="longitude", method="linear")
gphh = xr.where(gphh < 10, 10, gphh)


wiw=wiw.sortby('time')
wiwh=wiw.interp(time=new_time, method='linear')
wiwh = wiwh.interpolate_na(dim="longitude", method="linear")

pbl=pbl.sortby('time')
pblh=pbl.interp(time=new_time, method='linear')

ustar=ustar.sortby('time')
ustarh=ustar.interp(time=new_time, method='linear')
dt24=dt24.sortby("time")
dt24h=dt24.interp(time=new_time, method='linear')
t2m=t2m.sortby('time')
t2mh=t2m.interp(time=new_time, method='linear')
sw=sw.sortby('time')
swh=sw.interp(time=new_time, method='linear')

lsp=lsp.sortby('time')
lsph=lsp.interp(time=new_time, method='linear')
dlsph=lsph.diff("time")
zero_drain = xr.zeros_like(lsph.isel(time=0))
drain = xr.concat([zero_drain, dlsph], dim="time")
#drain.to_netcdf('/mnt/g/SAND3D/drain.nc')
drain["time"] = lsph.time
drain=drain.transpose('time', 'latitude', 'longitude')
drain=xr.where(drain<0,0,drain)
wetscanv=drain*1000*8.0*10e-3

gph_0=gph_0.sortby('time')
gph_0h=gph_0.interp(time=new_time, method='linear')
gph_bottom=gph_bottom.interp(time=new_time, method='linear')

height_above_ground = gph_0h - gph_bottom
#height_above_ground.to_netcdf('/mnt/g/SAND3D/height.nc')
pbl_mask = height_above_ground < pblh.broadcast_like(height_above_ground)
pbl_mask2=height_above_ground < pblh.broadcast_like(height_above_ground)/2.0
pbl_mask3=height_above_ground < 400.
kvh1 = xr.where(pbl_mask, 0.2/1200., 0.001/1200.)
kvh = xr.where(pbl_mask2, 0.5/600., kvh1)
kvh = xr.where(pbl_mask3, 0.7/180., kvh)
kvh = kvh.transpose('time', 'sigma', 'latitude', 'longitude')
#kvh.to_netcdf('/mnt/g/SAND3D/kv.nc')

ntime=121
nlat, nlon=61, 81
ccst_data = np.zeros((ntime, 13,nlat, nlon)) 
fcst_data = np.zeros((ntime, 13,nlat, nlon))   

sandort_data=np.zeros((ntime, nlat, nlon))
ccsa_data=np.zeros((ntime,n_labels,nlat, nlon))
#vd_data = np.zeros((ntime, 13,nlat, nlon))   
#wdep_data = np.zeros((ntime, nlat, nlon)) 
#ddep_data = np.zeros((ntime, nlat, nlon)) 
dx_cp = cp.asarray(dx, dtype=cp.float64)
dy_cp = cp.asarray(dy, dtype=cp.float64)
ccr_cp=  cp.asarray(ccr, dtype=cp.float64)
# 一步生成 ccs_by_label
labels_mask = (region_labels_cp == cp.arange(n_labels)[:, None, None])  # (n_labels, nlat, nlon)
dei_per_label_cp = cp.broadcast_to(
    labels_mask[:, None, :, :],  # (n_labels, 1, nlat, nlon)
    (n_labels, 13, nlat, nlon)
).astype(cp.float64)

# xr.DataArray(
#     cp.asnumpy(dei_by_label),
#     dims=('label', 'sigma', 'latitude', 'longitude'),
#     coords={'label': unique_labels, 'sigma':gphh.sigma.data, "latitude": gphh.latitude.data, "longitude":gphh.longitude.data}
# ).to_dataset(name='dei').to_netcdf('dei_by_label.nc')

ccs_by_label = cp.zeros((n_labels, 13,nlat, nlon), dtype=cp.float64)

for t_idx, timetip in enumerate(new_time):
    print(timetip)
    day_of_year = timenow.timetuple().tm_yday
    uh=wiuh.sel(time=timetip)
    vh=wivh.sel(time=timetip)
    wh=wiwh.sel(time=timetip)
    gphh1=gphh.sel(time=timetip)
    gph_0h1=height_above_ground.sel(time=timetip)
    wets=wetscanv.sel(time=timetip)
    dth=dt24h.sel(time=timetip)
    t2h=t2mh.sel(time=timetip)
    swhh=swh.sel(time=timetip)
    sandor=xr.where(t2h>270.,sandor,sandinitiallow*0.95)
    if(day_of_year<160):
        sandor=xr.where((t2h>276.)&(sandor<sandinitiallow),sandinitialhigh,sandor)
    sandor=xr.where(swhh<0.05,sandor,sandinitialhigh)
    deicor=emisc['pm10e'].sel(time=timetip)
    deicor=xr.where(np.isnan(deicor),1.0,deicor)
    dei=emisc['pm10e'].sel(time=timetip).values
    dei=dei*(sandor.values)

    sandor=sandor-(sandinitialhigh-sandinitiallow)*(deicor/(emisssum*4.0))
    sandor=xr.where(np.isnan(sandor),1.0,sandor)
    sandor= xr.where(sandor>sandinitiallow,sandor,sandinitiallow)
    kv1h=kvh.sel(time=timetip)
    ustarh1=ustarh.sel(time=timetip)
    dei = np.where(np.isnan(dei), 0, dei)
    dei = np.where(dei<0, 0, dei)*1e9
    gphh1['sigma']=uh.sigma
    dei=xr.DataArray(data=dei,dims=('latitude', 'longitude'),coords={"latitude": gphh1.latitude.data, "longitude":gphh1.longitude.data})
    if timetip == timenow:
        #break
        ccsi=ccsy
        ccsi=xr.DataArray(data=ccsi.data,dims=('sigma','latitude', 'longitude'),coords={'sigma':gphh1.sigma.data,"latitude": gphh1.latitude.data, "longitude":gphh1.longitude.data})   
        ccsi_np = ccsi.values  # xarray -> numpy
        ccsi_cp = cp.asarray(ccsi_np, dtype=cp.float64)
        fcsi=fcsy
        fcsi=xr.DataArray(data=fcsi.data,dims=('sigma','latitude', 'longitude'),coords={'sigma':gphh1.sigma.data,"latitude": gphh1.latitude.data, "longitude":gphh1.longitude.data})   
        fcsi_np = fcsi.values  # xarray -> numpy
        fcsi_cp = cp.asarray(fcsi_np, dtype=cp.float64)
        ccssai_cp=cp.asarray(ccsay, dtype=cp.float64)
    e_full = xr.zeros_like(ccsi)
    uh_cp = cp.asarray(uh.values, dtype=cp.float64)
    vh_cp = cp.asarray(vh.values, dtype=cp.float64)
    wh_cp = cp.asarray(wh.values, dtype=cp.float64)
    gphh1_cp = cp.asarray(gphh1.values, dtype=cp.float64)
    kv1h_cp = cp.asarray(kv1h.values, dtype=cp.float64)
    wets_cp = cp.asarray(wets.values, dtype=cp.float64)
    dei_cp=cp.zeros((n_labels, 13,nlat, nlon), dtype=cp.float64)
    dc_dt_h_sa=cp.zeros((n_labels, 13,nlat, nlon), dtype=cp.float64)
    # 创建条件掩码
    mask_dth_gt_min1 = dth > -1
    mask_dth_le_min3 = dth <= -5
    mask_dth_le_min1 = (dth <= 0)&(dth > -5)
    vhs=vh.sel(sigma=0.995)
    # 构建条件表达式
    condition0 = (gph_0h1 < 7000.) & (dth>7.)
    condition1 = (gph_0h1 < 5000.) & (vhs>=-1.) & mask_dth_gt_min1
    condition2 = (gph_0h1 < 2000.) &  mask_dth_le_min1
    condition3 = (gph_0h1 < 1000.) & mask_dth_le_min3
    condition4 = (gph_0h1 < 6000.) & (t2h>293.)
    condition5 = (gph_0h1 < 2000.) & (vhs<-1.) & mask_dth_gt_min1
    condition6 = (gph_0h1 < 1000.) & (vhs<-8.)
    # 计算efull的值
    e_full = xr.where(
        condition0,(1/7000),
        xr.where(
            condition6,(1/1000),
            xr.where(
                condition4,(1/6000),
                xr.where(
                    condition5,(1/2000.),
                    xr.where(
                        condition1,(1 / 5000.) ,
                        xr.where(
                            condition2,(1 / 2000.) ,
                            xr.where(
                                condition3,(1/1000.), 0  # 不满足任何条件时设为0
                                    )
                                )
                            )
                        )
                    )
                )
            )   
        
    e_full=xr.where(((dth>2)|(vhs>0.))&(soiltype!=12.),e_full*0.1,e_full)
    e_full=xr.where(((dth>-2)&(dth<=2.)&(vhs>-2.)&(soiltype!=12.)),e_full*0.5,e_full)
    e_full=xr.where(((vhs<-8.)&(soiltype!=12.)),e_full*1.2,e_full)
    e_full=xr.where(((soiltype==12.)),e_full*5.0,e_full)    
    e_full = e_full.transpose('sigma', 'latitude', 'longitude')
    dt=180
    nstep=int(3600/dt)
    factor = xr.zeros_like(ccsi)
    factor2=xr.zeros_like(ccsi)
    depshf=0.01*np.exp(-5.46*(ustarh1-0.60)**2)#ustarh1*ustarh1*0.2055#-0.0005
    depshf=xr.where(depshf<0.00001,0.00001,depshf)
    depshf=xr.where((ustarh1>0.6) ,depshf*0.1,depshf)
    depshf=xr.where((ustarh1>0.9) & (dth<-6)&(vhs<-1.),depshf*0.01,depshf)    
    depshf=xr.where( (dth<-8.)&(vhs<-1.),depshf*0.001,depshf)
    depshf=xr.where((soiltype==1.0)& (dth<3.),depshf*0.001,depshf)
    depshf=xr.where(soiltype==1.0,depshf*0.00001,depshf)
    factor.loc[{'sigma': 0.995}]=depshf


    factor_cp = cp.asarray(factor.values, dtype=cp.float64)
    dei_cp = cp.asarray(dei, dtype=cp.float64) 
    

    for i in range(nstep):
        ccssai_cp_total = cp.sum(ccssai_cp, axis=0, keepdims=True)  # shape: (1, 13, nlat, nlon)
        ccssai_cp_total = cp.where(ccssai_cp_total == 0, 1.0, ccssai_cp_total)
        ccssai_cp_proportions = ccssai_cp / ccssai_cp_total
        dc_dt_h,dc_dt_h_sa = calculate_horizontal_divergence(uh_cp, vh_cp, ccsi_cp, gphh1_cp, dx_cp, dy_cp, dt,ccssai_cp_proportions)
        dc_dt_vd ,dc_dt_vd_sa= calculate_vertical_divergence(wh_cp, ccsi_cp, gphh1_cp,ccssai_cp_proportions)
        dc_dt_vdf ,dc_dt_vdf_sa= calculate_vertical_diffusion(wh_cp, ccsi_cp, kv1h_cp, gphh1_cp,ccssai_cp_proportions)
        dc_dt_dei=dei_cp*ccr_cp*cp.asarray(e_full, dtype=cp.float64)

        # dc_dt_h_f,feastf,fsouthf = calculate_horizontal_divergence(uh_cp, vh_cp, fcsi_cp, gphh1_cp, dx_cp, dy_cp, dt)
        # dc_dt_vd_f = calculate_vertical_divergence(wh_cp, fcsi_cp, gphh1_cp)
        # dc_dt_vdf_f = calculate_vertical_diffusion(wh_cp, fcsi_cp, kv1h_cp, gphh1_cp)

        #print(ccssai_cp_proportions.shape)
        dc_dt = (
            (dc_dt_h) +
            dc_dt_dei +
            (dc_dt_vd) +
            (dc_dt_vdf) -
            ccsi_cp * (wets_cp) * 0.05 -
            ccsi_cp  * (factor_cp) / (gphh1_cp) 
        )
        # dc_dt_f = (
        #     (dc_dt_h_f) +
        #     dei_cp*(1-ccr_cp)*cp.asarray(e_full, dtype=cp.float64) +
        #     (dc_dt_vd_f) +
        #     (dc_dt_vdf_f) -
        #     fcsi_cp * (wets_cp) * 0.02-
        #     fcsi_cp * (factor2_cp) / (gphh1_cp)
        # )

        ccsi_cp = ccsi_cp + dc_dt * dt
        ccsi_cp = cp.maximum(ccsi_cp, 0)
        # fcsi_cp = fcsi_cp + dc_dt_f * dt
        # fcsi_cp = cp.maximum(fcsi_cp, 0)
        dei_sa_cp = dei_cp*ccr_cp*cp.asarray(e_full, dtype=cp.float64)*dei_per_label_cp
        ccssai_cp=ccssai_cp+dei_sa_cp*dt+dc_dt_h_sa*dt+dc_dt_vd_sa*dt#+dc_dt_vdf_sa*dt#+(dc_dt-dc_dt_h-dc_dt_dei-dc_dt_vd-dc_dt_vdf)*ccssai_cp_proportions*dt
        ccssai_cp = cp.maximum(ccssai_cp, 0)
        ccssai_cp_total = cp.sum(ccssai_cp, axis=0, keepdims=True)  # shape: (1, 13, nlat, nlon)
        ccssai_cp_total = cp.where(ccssai_cp_total == 0, 1.0, ccssai_cp_total)
        ccssai_cp_proportions = ccssai_cp / ccssai_cp_total
        ccssai_cp=ccsi_cp*ccssai_cp_proportions


    ccst_data[t_idx,:, :, :]=cp.asnumpy(ccsi_cp)
#    fcst_data[t_idx,:, :, :]=cp.asnumpy(fcsi_cp)
    ccsa_data[t_idx,:,:, :]=cp.asnumpy(ccssai_cp[:,-1,:,:])
    sandort_data[t_idx,:, :]=sandor.data

lats = np.arange(15, 55.2, 0.25)
lons = np.arange(70, 140.2, 0.25)

ds = xr.Dataset({
    "ccs":(['time', 'sigma', 'latitude', 'longitude'],ccst_data),
#    "fcs":(['time', 'sigma', 'latitude', 'longitude'],fcst_data),
    "sandor":(['time', 'latitude', 'longitude'],sandort_data),
#    "wdep":(['time', 'latitude', 'longitude'],wdep_data),
#    "ddep":(['time', 'latitude', 'longitude'],ddep_data),      
    }, coords={"time": gphh.time.data,  "sigma": gphh.sigma.data,"latitude": gphh.latitude.data, "longitude":gphh.longitude.data,})
ds['ccs'] = ds['ccs'].astype(np.float32)
#ds['fcs'] = ds['fcs'].astype(np.float32)
ds['sandor']=ds['sandor'].astype(np.float32)
ds.to_netcdf("/mnt/g/FQDUST/dust_forecast_"+init_time+"_phys-sigma-depbottom2-180s-wet-dt24-gtsi-met-bj.nc")
d2s2=xr.Dataset({
    "ccsa":([ 'time','lable', 'latitude', 'longitude'],ccsa_data)  
    }, coords={"time": gphh.time.data,  'label': unique_labels,"latitude": gphh.latitude.data, "longitude":gphh.longitude.data,})
d2s2['ccsa']=d2s2['ccsa'].astype(np.float32)
d2s2.to_netcdf("/mnt/g/FQDUST/dust_forecast_"+init_time+"_phys-bjsa.nc")