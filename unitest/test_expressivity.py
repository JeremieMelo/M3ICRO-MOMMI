import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.fft
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyutils.plot import batch_plot, set_axes_size_ratio
from pyutils.torch_train import load_model
from torch import nn

from core.models import DPE
from core.models.layers import DPELinear

SMALL_SIZE = 16
MEDIUM_SIZE = 16
BIGGER_SIZE = 16
color_dict = {
    "black": "#000000",
    "red": "#de425b",  # red
    "blue": "#1F77B4",  # blue
    "orange": "#f58055",  # orange
    "yellow": "#f6df7f",  # yellow
    "green": "#2a9a2a",  # green
    "grey": "#979797",  # grey
    "purple": "#AF69C5",  # purple,
    "mitred": "#A31F34",  # mit red
    "pink": "#CDA2BE",
}

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def test_usv_diff(n_pads=5, n_ports=5, N=2000, dpe_ckpt="", mode="usv", multiplier=2, plot=False, w_bit: int = 4096):
    device = torch.device("cuda:0")
    # p, q = 100, 100
    p, q = 200, 200
    # p, q = 400, 400
    # x [bs, q, k]

    dpe = DPE(
        n_pads=n_pads,
        n_ports=n_ports,
        act_cfg=dict(type="GELU"),
        hidden_dims=[256, 256, 128, 128, 128],
        dropout=0,
        device=device,
    ).to(device)
    dpe.set_dpe_noise_ratio(0)
    dpe.requires_grad_(False)
    dpe.eval()
    load_model(dpe, dpe_ckpt)

    layer = DPELinear(
        p * n_ports,
        q * n_ports,
        n_pads=n_pads,
        mini_block=n_ports,
        bias=False,
        dpe=dpe.forward,
        w_bit=w_bit,
        mode=mode,
        path_multiplier=multiplier,
        sigma_trainable="row_col",
        device=device,
    ).to(device)
    # ususv x 2 = 0.2585
    # usv x 3 = 0.4630
    layer.reset_parameters(mmi_init=True)
    # target = torch.randn(p, q, k, k, dtype=torch.cfloat, device=device).svd()[0]
    target = torch.randn(p, q, n_ports, n_ports, device=device)
    # target = torch.randn(p, q, k, k, device=device).svd()[0]
    target_norm = target.square().sum(dim=(-1, -2))
    # optimizer = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=1e-2)
    if w_bit < 16:
        lr = 1e-5
    elif w_bit <= 256:
        lr = 5e-6 * w_bit
    else:
        lr = 2e-3
    optimizer = torch.optim.Adam(
        [{"params": layer._fc_pos.weight, "lr": lr}, {"params": layer._fc_pos.sigma, "lr": 1e-2}], lr=3e-4
    )
    # optimizer = torch.optim.Adam([layer._fc_pos.weight], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N, last_epoch=-1)
    for i in range(N):
        weight = layer._weight[0]  # [p, q, k, k] complex
        # weight_p, weight_n = weight.chunk(2, dim=0)
        # weight = weight_p + weight_n  # [p/2, q, k, k] complex

        weight = weight.reshape(
            [
                layer.path_multiplier,
                weight.shape[0] // layer.path_multiplier,
                weight.shape[1],
                weight.shape[2],
                weight.shape[3],
            ]
        ).sum(0)

        weight = torch.view_as_real(weight)  # [p/2, q, k, k, 2] real
        weight = weight.permute(0, 4, 1, 2, 3).flatten(0, 1)  # [p,q,k,k] real
        # total_loss = target.mul(weight.conj()).real.sum(dim=(-1, -2)) / n_ports
        # loss = -total_loss.mean()
        total_loss = weight.sub(target).square().sum(dim=(-1, -2)) / target_norm
        loss = torch.nn.functional.mse_loss(weight, target)
        # loss = total_loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 1000 == 0 or i == N - 1:
            print(f"Step: {i}, loss={loss.item():.4f}, rel dist={total_loss.mean().item():.4f}")
            # print(layer._fc_pos.weight.grad[0,0,0])
            # print(layer._fc_pos.weight[0,0,0])
            # print(layer._fc_pos.sigma.grad[0,0])
            # print(layer._fc_pos.sigma[0,0])
    fidelity = 1 - total_loss.mean().item()
    # x = total_loss.data.cpu().flatten().numpy()
    # weight = torch.view_as_real(weight)
    # w = torch.view_as_real(w.data)
    # x = weight.sub(w).square().sum(dim=(-1,-2, -3)) / weight.square().sum(dim=(-1, -2, -3))
    # x = x.data.cpu().numpy()

    if plot:
        x = 1 - total_loss.data.cpu().flatten().numpy()
        fig, ax = plt.subplots(1, 1)
        n, bins, patches = ax.hist(x, bins=30, density=True, facecolor="b", alpha=0.75, rwidth=0.8)

        plt.xlabel("Fidelity")
        plt.ylabel("Count")
        plt.title("Histogram of MMI Unitary")
        plt.text(60, 0.025, r"$\mu=100,\ \sigma=15$")
        # plt.xlim(0.9, 1)  # 4
        # plt.xlim(0.85, 0.9) # 5
        plt.xlim(0.35, 0.40)  # 10
        # plt.ylim(0, 70 * p / 100)
        plt.ylim(0, np.max(n) * 1.2)
        plt.grid(False)
        frame = plt.gca()
        frame.axes.yaxis.set_ticklabels([])
        set_axes_size_ratio(1, 1.1, fig=fig, ax=ax)

        # plt.tight_layout()
        plt.savefig(f"./figs/MMIONN_Unitary_hist_{n_ports}_USVDiff.png", dpi=300)
        plt.close()
    return fidelity


def compare_dpe_and_eme(n_pads=4, n_ports=4, inport_idx=0, outport_idx=0, pad_idx=3, dpe_ckpt="", n_level=8):
    device = torch.device("cuda:0")
    # x [bs, q, k]
    # inport_idx = 0
    # outport_idx = 0
    # pad_idx = 3

    dpe = DPE(
        n_pads=n_pads,
        n_ports=n_ports,
        act_cfg=dict(type="GELU"),
        hidden_dims=[256, 256, 128, 128, 128],
        dropout=0,
        device=device,
    ).to(device)
    dpe.set_dpe_noise_ratio(0)
    dpe.requires_grad_(False)
    dpe.eval()
    load_model(dpe, dpe_ckpt)

    train_data_path = "./data/mmi/port_4_res_8_range_0.03/training.pt"
    test_data_path = "./data/mmi/port_4_res_8_range_0.03/test.pt"
    train_data = torch.load(train_data_path)
    test_data = torch.load(test_data_path)
    train_data, train_target = train_data["data"][0], torch.view_as_real(train_data["data"][1]).flatten(1)
    test_data, test_target = test_data["data"][0], torch.view_as_real(test_data["data"][1]).flatten(1)
    data, target = torch.cat([train_data, test_data], 0), torch.cat([train_target, test_target], 0)
    idx = torch.argsort(data[:, 0] * 1000 + data[:, 1] * 100 + data[:, 2] * 10 + data[:, 3] * 1)
    data = data[idx]
    target = target[idx]

    # all dataset prediction
    all_pred = torch.view_as_real(dpe(data.to(device))).flatten(1).cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 4.5))
    im = ax.matshow(np.abs(target.numpy() - all_pred), aspect="auto", cmap="jet")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    plt.tight_layout()

    # ax[1].matshow(all_pred, aspect="auto")
    name = "CompareEME_DPE"
    fig.savefig(f"./figs/{name}.png", dpi=300)

    # print(data[5])
    # print(target[5])

    target = target.reshape(n_level, n_level, n_level, n_level, -1)
    if pad_idx == 0:
        target = target.permute(1, 2, 3, 0, 4)
    elif pad_idx == 1:
        target = target.permute(0, 2, 3, 1, 4)
    elif pad_idx == 2:
        target = target.permute(0, 1, 3, 2, 4)
    target = target[0, 0, 0, :, (outport_idx * n_ports + inport_idx) * 2 : (outport_idx * n_ports + inport_idx) * 2 + 2]
    # target = target[
    #     8 ** (3-pad_idx) - 1 : 8 ** (3-pad_idx) + 7,
    #     (outport_idx * n_ports + inport_idx) * 2 : (outport_idx * n_ports + inport_idx) * 2 + 2,
    # ]  # [8, 2] real
    # data = data[
    #     8 ** (3-pad_idx) - 1 : 8 ** (3-pad_idx) + 7,
    #     -1,
    # ]  # [4096, 4] [8] real
    data = np.linspace(0, 1, 8)

    sweep_x = torch.linspace(0, 1, 200)
    inputs = torch.cat([torch.zeros(200, pad_idx), sweep_x.unsqueeze(1), torch.zeros(200, 3 - pad_idx)], 1)
    pred = torch.view_as_real(dpe(inputs.to(device))).flatten(1).cpu().numpy()
    pred = pred[:, (outport_idx * n_ports + inport_idx) * 2 : (outport_idx * n_ports + inport_idx) * 2 + 2]

    black, blue, orange, purple, green, red, pink = (
        color_dict["black"],
        color_dict["blue"],
        color_dict["orange"],
        color_dict["purple"],
        color_dict["green"],
        color_dict["mitred"],
        color_dict["pink"],
    )

    fig, ax = plt.subplots(1, 1)
    name = "CompareEME_DPE"
    print(target.shape, data.shape)
    ax.scatter(
        data,
        target[:, 0],
        c=black,
        s=4,
        alpha=1,
    )  # real
    ax.scatter(
        data,
        target[:, 1],
        c=blue,
        s=4,
        alpha=1,
    )  # imag
    ax.plot(
        sweep_x.numpy(),
        pred[:, 0],
        c=black,
    )
    ax.plot(
        sweep_x.numpy(),
        pred[:, 1],
        c=blue,
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.55, 0.55)
    set_axes_size_ratio(0.6, 0.6, fig, ax)
    plt.savefig(f"./figs/{name}_in-{inport_idx}_out-{outport_idx}_pad-{pad_idx}.png", dpi=300)


def plot_ptc_expressivity(data, n_ports=4, n_multiplier=7, n_cascade=4, name="ComparePTCFidelity"):
    data = [x.split(" ") for x in data.split("\n")]
    fidelity = np.array([float(i[2]) for i in data]).reshape(n_multiplier, n_cascade)
    fig, ax = None, None
    name = f"{name}_{n_ports}"
    y = np.arange(1, n_multiplier + 1)
    x = np.arange(1, n_cascade + 1)
    fig, ax, _ = batch_plot(
        "none",
        raw_data={"x": x, "y": y, "z": fidelity},
        name=name,
        xlabel=r"\#Cascade",
        ylabel=r"\#Path",
        fig=fig,
        ax=ax,
        xrange=[1, n_cascade + 1.01, 1],
        yrange=[1, n_multiplier + 1.01, 1],
        xlimit=[0.5, n_cascade + 0.5],
        ylimit=[0.5, n_multiplier + 0.5],
        xformat="%.0f",
        yformat="%.0f",
        figscale=[1.65, 1.58],
        fontsize=12,
        linewidth=1,
        gridwidth=0.5,
        ieee=True,
    )

    for i in x:
        for j in y:
            z = fidelity[j - 1, i - 1]
            ax.annotate(f"{z:.3f}", xy=(i, j), xytext=(i, j), ha="center", va="center", fontsize=11)
    dx = (x[-1] - x[0]) / (len(x) - 1)
    dy = (y[-1] - y[0]) / (len(y) - 1)
    x = np.arange(x[0] - dx / 2, x[-1] + 1.1 * dx / 2, dx)
    y = np.arange(y[0] - dy / 2, y[-1] + 1.1 * dy / 2, dy)

    im = ax.pcolormesh(
        x,
        y,
        (fidelity + 1e-12) ** 0.25,
        vmin=0.75,
        vmax=1,
        shading="auto",
        cmap=plt.cm.RdYlGn,
        linewidth=0.5,
        alpha=0.8,
    )
    plt.rcParams.update(
        {
            # "font.family": "serif",
            # "font.sans-serif": ["Helvetica"]
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
        }
    )
    plt.yticks(fontname="Arial")
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xlim(0.5, n_cascade + 0.5)
    ax.set_ylim(0.5, n_multiplier + 0.5)
    plt.minorticks_off()
    ax.tick_params(axis="both", which="both", length=0)
    set_axes_size_ratio(0.6, 1.3 * 0.6, fig=fig, ax=ax)
    ax.invert_yaxis()

    fig.savefig(f"./figs/{name}.png")
    # fig.savefig(f"./figs/{name}.svg")
    # fig.savefig(f"./figs/{name}.pdf")
    # pdf_crop(f"./figs/{name}.pdf", f"./figs/{name}.pdf")

    # print(data)


def test_mommi_robust(
    n_pads=5, n_ports=5, N=2000, dpe_ckpt="", mode="usv", multiplier=2, plot=False, w_bit: int = 4096
):
    device = torch.device("cuda:0")
    # p, q = 100, 100
    p, q = 200, 200
    # p, q = 400, 400
    # x [bs, q, k]

    dpe = DPE(
        n_pads=n_pads,
        n_ports=n_ports,
        act_cfg=dict(type="GELU"),
        hidden_dims=[256, 256, 128, 128, 128],
        dropout=0,
        device=device,
    ).to(device)
    dpe.set_dpe_noise_ratio(0)
    dpe.requires_grad_(False)
    dpe.eval()
    load_model(dpe, dpe_ckpt)

    layer = DPELinear(
        p * n_ports,
        q * n_ports,
        n_pads=n_pads,
        mini_block=n_ports,
        bias=False,
        dpe=dpe.forward,
        w_bit=w_bit,
        mode=mode,
        path_multiplier=multiplier,
        sigma_trainable="row_col",
        device=device,
    ).to(device)
    # ususv x 2 = 0.2585
    # usv x 3 = 0.4630
    layer.reset_parameters(mmi_init=True)
    # target = torch.randn(p, q, k, k, dtype=torch.cfloat, device=device).svd()[0]
    target = layer._weight[0]  # [p, q, k, k] complex
    target = torch.view_as_real(
        target.reshape(
            [
                layer.path_multiplier,
                target.shape[0] // layer.path_multiplier,
                target.shape[1],
                target.shape[2],
                target.shape[3],
            ]
        ).mean(0)
    )  # [p/2, q, k, k, 2] real
    target = target.permute(0, 4, 1, 2, 3).flatten(0, 1)  # [p, q, k, k] real
    # target = torch.randn(p, q, k, k, device=device).svd()[0]
    target_norm = target.square().sum(dim=(-1, -2))

    for i, noise in enumerate(np.arange(0, 0.0101 / 2, 0.001 / 2)):
        dpe.set_pad_noise(noise)
        weight = layer._weight[0]  # [p, q, k, k] complex
        # weight_p, weight_n = weight.chunk(2, dim=0)
        # weight = weight_p + weight_n  # [p/2, q, k, k] complex

        weight = torch.view_as_real(
            weight.reshape(
                [
                    layer.path_multiplier,
                    weight.shape[0] // layer.path_multiplier,
                    weight.shape[1],
                    weight.shape[2],
                    weight.shape[3],
                ]
            ).mean(0)
        )

        weight = weight.permute(0, 4, 1, 2, 3).flatten(0, 1)  # [p,q,k,k] real
        # total_loss = target.mul(weight.conj()).real.sum(dim=(-1, -2)) / n_ports
        # loss = -total_loss.mean()
        total_loss = weight.sub(target).square().sum(dim=(-1, -2)) / target_norm
        err_mean, err_std = total_loss.mean().item(), total_loss.std().item()
        # loss = total_loss.mean()

        # print(f"Noise: {noise}, err_mean = {err_mean:.4f}, err_std = {err_std:.4f}")
        print(err_mean, err_std)

    if plot:
        x = 1 - total_loss.data.cpu().flatten().numpy()
        fig, ax = plt.subplots(1, 1)
        n, bins, patches = ax.hist(x, bins=30, density=True, facecolor="b", alpha=0.75, rwidth=0.8)

        plt.xlabel("Fidelity")
        plt.ylabel("Count")
        plt.title("Histogram of MMI Unitary")
        plt.text(60, 0.025, r"$\mu=100,\ \sigma=15$")
        # plt.xlim(0.9, 1)  # 4
        # plt.xlim(0.85, 0.9) # 5
        plt.xlim(0.35, 0.40)  # 10
        # plt.ylim(0, 70 * p / 100)
        plt.ylim(0, np.max(n) * 1.2)
        plt.grid(False)
        frame = plt.gca()
        frame.axes.yaxis.set_ticklabels([])
        set_axes_size_ratio(1, 1.1, fig=fig, ax=ax)

        # plt.tight_layout()
        plt.savefig(f"./figs/MMIONN_Unitary_hist_{n_ports}_USVDiff.png", dpi=300)
        plt.close()
    return


def test_fftonn_robust(k=4, N=5000):
    P, Q = 200, 200
    device = torch.device("cuda:0")

    x = torch.eye(k, k, dtype=torch.cfloat, device=device)
    sigma = torch.randn(P // 2, Q, k, dtype=torch.cfloat, device=device)
    # x [bs, q, k]

    from core.models.butterfly_utils import BatchTrainableButterfly

    layer_b = BatchTrainableButterfly(
        batch_size=(P // 2, Q),
        length=k,
        device=device,
        bit_reversal=False,
    )
    layer_p = BatchTrainableButterfly(
        batch_size=(P // 2, Q),
        length=k,
        device=device,
        mode="full_reverse",
        bit_reversal=False,
    )

    # b = torch.view_as_complex(layer_b(x))  # [p/2, q, k, k]
    # p = torch.view_as_complex(layer_p(x))  # [p/2, q, k, k]
    b = layer_b(x)
    p = layer_p(x)
    print(b.shape, p.shape, sigma.shape)
    weight = p.matmul(sigma.unsqueeze(-1).mul(b))  # [p/2, q, k, k] complex
    weight = torch.view_as_real(weight)  # [p/2, q, k, k, 2] real
    target = weight.permute(0, 4, 1, 2, 3).flatten(0, 1)  # [p,q,k,k] real
    target_norm = target.square().sum(dim=(-1, -2))

    for i, noise in enumerate(np.arange(0, 0.0101 / 2, 0.001 / 2)):
        layer_b.set_phase_noise(noise * np.pi * 2)
        layer_p.set_phase_noise(noise * np.pi * 2)
        b = layer_b(x)  # [p/2, q, k, k]
        p = layer_p(x)  # [p/2, q, k, k]
        weight = p.matmul(sigma.unsqueeze(-1).mul(b))  # [p/2, q, k, k] complex
        weight = torch.view_as_real(weight)  # [p/2, q, k, k, 2] real
        weight = weight.permute(0, 4, 1, 2, 3).flatten(0, 1)  # [p,q,k,k] real
        # total_loss = target.mul(weight.conj()).real.sum(dim=(-1, -2)) / n_ports
        # loss = -total_loss.mean()
        total_loss = weight.sub(target).square().sum(dim=(-1, -2)) / target_norm
        loss = torch.nn.functional.mse_loss(weight, target)
        # loss = total_loss.mean()
        total_loss = weight.sub(target).square().sum(dim=(-1, -2)) / target_norm
        err_mean, err_std = total_loss.mean().item(), total_loss.std().item()
        # loss = total_loss.mean()

        # print(f"Noise: {noise}, err_mean = {err_mean:.4f}, err_std = {err_std:.4f}")
        print(err_mean, err_std)


if __name__ == "__main__":
    p = 4
    k = 4
    N = 5000
    dpe_ckpt4 = "./checkpoint/mmi/dpe/pretrain/MMI_port_4_pad_4_res_8_range_0.03.pt"

    fid = test_usv_diff(n_ports=p, n_pads=k, N=N, dpe_ckpt=dpe_ckpt4, multiplier=2, mode="usv")
    print(f"Port = {p}, Pad = {k}")
    print(fid[0], fid[1], fid[2])

    p = 5
    k = 5
    dpe_ckpt5 = "./checkpoint/mmi/dpe/pretrain/MMI_port_5_pad_5_res_6_range_0.03.pt"
    fid = test_usv_diff(n_ports=p, n_pads=k, N=N, dpe_ckpt=dpe_ckpt5, multiplier=2, mode="usv")
    print(f"Port = {p}, Pad = {k}")
    print(fid[0], fid[1], fid[2])

    p = 10
    k = 5
    dpe_ckpt10 = "./checkpoint/mmi/dpe/pretrain/MMI_port_10_pad_5_res_6_range_0.03.pt"
    fid = test_usv_diff(n_ports=p, n_pads=k, N=N, dpe_ckpt=dpe_ckpt10, multiplier=2, mode="ususv")
    print(f"Port = {p}, Pad = {k}")
    print(fid[0], fid[1], fid[2])
