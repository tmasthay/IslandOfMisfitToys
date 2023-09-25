import matplotlib.pyplot as plt


def make_plots(*, v_true, v_init, vp):
    v = vp()
    vmin = v_true.min()
    vmax = v_true.max()
    _, ax = plt.subplots(3, figsize=(10.5, 10.5), sharex=True, sharey=True)
    ax[0].imshow(
        v_init.cpu().T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax
    )
    ax[0].set_title("Initial")
    ax[1].imshow(
        v.detach().cpu().T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax
    )
    ax[1].set_title("Out")
    ax[2].imshow(
        v_true.cpu().T, aspect='auto', cmap='gray', vmin=vmin, vmax=vmax
    )
    ax[2].set_title("True")
    plt.tight_layout()
    plt.savefig('example_distributed_ddp.jpg')

    v.detach().cpu().numpy().tofile('marmousi_v_inv.bin')
