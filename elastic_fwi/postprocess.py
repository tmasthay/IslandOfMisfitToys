import torch
import matplotlib.pyplot as plt
import os

def go():
    res = torch.load('forward.pt')
    ref = res[res.shape[0] // 2][res.shape[1] // 2]
    u = torch.empty(res.shape[0], res.shape[1])
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            u[i][j] = 1e10 * torch.norm( res[i][j] - ref )**2
    plt.imshow(u)
    plt.title(r'L^2')
    plt.colorbar()
    plt.savefig('L2.jpg')
    return u

def show_data(filename, base_name):
    res = torch.load(filename)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            plt.imshow(res[i,j,0,:,:], cmap='seismic', aspect='auto')
            plt.title('Receiver Data')
            plt.xlabel('Receiver No')
            plt.ylabel('Time (s)')
            plt.colorbar()
            plt.savefig(f'{base_name}_{i}_{j}.jpg') 
            plt.clf()
            print(f'{base_name},{i},{j}') 

def show_sources(step_size):
    filename = 'sources.pt'
    base_name = filename.replace('.pt','')
    res = torch.load(filename)
    for ix in range(res.shape[0]):
        for iy in range(res.shape[1]):
            for it in range(0,res.shape[-1],step_size): 
                plt.imshow(res[ix,iy,0,:,:,it], cmap='seismic', aspect='auto')
                plt.title('Receiver Data')
                plt.xlabel('Receiver No')
                plt.ylabel('Time (s)')
                plt.colorbar()
                plt.savefig(f'{base_name}_{ix}_{iy}_{it}.jpg') 
                plt.clf()
                print(f'{base_name},{ix},{iy},{it}')
    os.system(f'convert delay -20 -loop 0 {base_name}*.jpg {base_name}.gif')
    os.system(f'rm {base_name}*.jpg')

if( __name__ == "__main__" ):
    show_sources(100)
