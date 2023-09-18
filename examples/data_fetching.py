from misfit_toys.utils import *
import matplotlib.pyplot as plt
import os

# subset = 'marmousi marmousi2'
path  = 'conda/data/marmousi'

# u = fetch_and_convert_data(subset=subset, path=path)

vp = get_data2(field='vp', path=path)
obs_data = get_data2(field='obs_data', path=path)
rho = get_data2(field='rho', path=path)
vs = get_data2(field='vs', path=path, allow_none=True)
metadata = get_metadata(path=path)

# print(vp.shape)
# print(obs_data.shape)
# print(rho.shape)
# print(type(vs))
# print(metadata)
# # input(get_primitives(metadata))

prims = get_primitives(metadata)
nt = obs_data.shape[-1]
dt = prims['dt']
ny = vp.shape[0]
dy = prims['dy']
plot_extent = [0, ny*dy, nt*dt, 0]

plot_every = 10
true_path = parse_path(path)
fig_path = os.path.join(true_path, 'figs')
os.system(f'mkdir -p {fig_path}')

# vmin, vmax = obs_data.min(), obs_data.max()
vmin, vmax = None, None
cmap = 'jet'

def illuminate(y, dt, alpha=2.0, beta=1.0, gamma=None):
    nt = y.shape[0]
    t = torch.arange(nt, dtype=torch.float32) * dt
    if( gamma is None ):
        gamma = y.mean()
    u = gamma + beta * t**alpha
    for i in range(y.shape[0]):
        y[i] *= u[i]
    return y

for i in range(0, obs_data.shape[0], plot_every):
    print(f'Shot {i}')
    plt.imshow(
        obs_data[i].T, 
        aspect='auto', 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax,
        extent=plot_extent
    )
    plt.colorbar()
    plt.ylabel('Time (s)')
    plt.xlabel('Rec Loc (m)')
    plt.title(f'Shot {i}')
    curr = os.path.join(fig_path, f'obs_data_{i}.jpg')
    plt.savefig(curr)
    plt.clf()

    plt.imshow(
        illuminate(obs_data[i].T, dt, alpha=2.0, beta=1.0, gamma=2.0),
        aspect='auto', 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax,
        extent=plot_extent
    )
    plt.colorbar()
    plt.ylabel('Time (s)')
    plt.xlabel('Rec Loc (m)')
    plt.title(f'Shot {i}')
    curr = os.path.join(fig_path, f'illuminate_obs_data_{i}.jpg')
    plt.savefig(curr)
    plt.clf()

cmd = f'convert -delay 100 $(ls -tr {fig_path}/obs_data_*.jpg) ' + \
    f'{fig_path}/obs_data.gif'
print(f'Attempting gif -- cmd = "{cmd}"')
os.system(cmd)
print(f'Gif saved to {fig_path}/obs_data.gif')

cmd = f'convert -delay 100 $(ls -tr {fig_path}/illuminate_obs_data_*.jpg) ' + \
    f'{fig_path}/illuminate_obs_data.gif'
print(f'Attempting gif -- cmd = "{cmd}"')
os.system(cmd)
print(f'Gif saved to {fig_path}/obs_data.gif')

os.system(f'rm {fig_path}/*.jpg')