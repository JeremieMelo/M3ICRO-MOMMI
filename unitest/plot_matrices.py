from sklearn.manifold import TSNE
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from pyutils.plot import set_axes_size_ratio

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


def plot_matrices():
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

    num_matrices = target.shape[0]
    random_matrices = torch.view_as_real(torch.randn(1, 4, 4, dtype=torch.cfloat).svd()[0]).flatten(1)
 
    tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=3)
    total_matrices = torch.cat([target, random_matrices])
    total_matrices = target
    total_matrices = torch.view_as_complex(total_matrices.view(-1, 4, 4, 2))  # [b, 5, 5] complex


    print(total_matrices.det())
    total_matrices = torch.view_as_real(total_matrices).flatten(1)
    target_embedded = tsne.fit_transform(total_matrices).T

    c = data.sum(1).numpy()  # use the sum of permittivity as color encoder
    fig, ax = plt.subplots(1, 1)

    size = np.ones([target_embedded.shape[1]])
    plt.scatter(
        target_embedded[0, :num_matrices],
        target_embedded[1, :num_matrices],
        c=c,
        cmap=plt.cm.rainbow,
        s=size,
        alpha=0.5,
    )
    size = np.zeros([target_embedded.shape[1]])
    size[[0, 7, 63, 511, 4095]] = 30
    plt.scatter(
        target_embedded[0, :num_matrices],
        target_embedded[1, :num_matrices],
        c=c,
        cmap=plt.cm.rainbow,
        s=size,
        alpha=1,
        marker="*",
        linewidths=0.5,
        edgecolors="black",
    )
    plt.colorbar()

    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis("tight")
    set_axes_size_ratio(1, 1, fig, ax)

    plt.savefig("./figs/MMI_port_4_res_8_matrices_det2d.png", dpi=300)


def plot_area_compare():
    L_0 = 55.4
    W_0 = 4.8
    PS = 90 * 40
    Y = 1.8 * 1.3
    DC = 29.3 * 2.4  # 2x2 MMI
    CR = 7.4 * 7.4
    k = np.arange(4, 65)
    logk = np.floor(np.log2(k))

    area_mommi_log = (
        2 * logk * L_0 * W_0 * k**2 / 4**2 + 4 * k * (logk - 1) * (PS + Y) + 2 * k * Y + k * (k - 1) * CR
    )
    alpha = ((2**0.5 - 3**0.5 + (2 + 12 * k) ** 0.5) ** 2 - 3) / 18 / k
    alpha = 0.7
    P = np.round((1 + (1 + 6 * alpha * k) ** 0.5) / 3)
    C = np.ceil((1 + (1 + 6 * alpha * k) ** 0.5) / 3)
    area_mommi_expr = (
        P * C * L_0 * W_0 * k**2 / 4**2
        + 2 * k * P * (C - 1) * (PS + Y)
        + 2 * (P - 1) * k * Y
        + (P - 1) * k * (k - 1) * CR
    )
    area_mzi = k**2 * (3 * PS + 2 * DC)
    k0 = 4
    area_osnn4 = (
        np.ceil(k / k0) ** 2 * k0 * (np.round(np.log2(k0)) + 2) * DC
        + np.ceil(k / k0) ** 2 * k0 * (2 * np.round(np.log2(k0)) + 2) * PS
        + k * (np.ceil(k / k0) - 1) * 2 * Y
        + np.ceil(k / k0) ** 2 * 2 * CR
    )

    k0 = 8
    area_osnn8 = (
        np.ceil(k / k0) ** 2 * k0 * (np.round(np.log2(k0)) + 2) * DC
        + np.ceil(k / k0) ** 2 * k0 * (2 * np.round(np.log2(k0)) + 2) * PS
        + k * (np.ceil(k / k0) - 1) * 2 * Y
        + np.ceil(k / k0) ** 2 * 16 * CR
    )

    for i in area_mommi_log:
        print(i)
    print()
    for i in area_mommi_expr:
        print(i)
    print()
    for i in area_mzi:
        print(i)
    print()
    for i in area_osnn4:
        print(i)
    print()
    for i in area_osnn8:
        print(i)


def plot_IL_compare():
    MMI = 0.33
    PS = 0.04
    Y = 0.3
    DC = 0.33
    CR = 0.02
    k = np.arange(4, 65)
    logk = np.floor(np.log2(k))
    area_mommi_log = 2 * Y + logk * MMI + (logk - 1) * (2 * Y + PS) + 2 * (k - 1) * CR
    alpha = 0.7
    P = np.round((1 + (1 + 6 * alpha * k) ** 0.5) / 3)
    C = np.ceil((1 + (1 + 6 * alpha * k) ** 0.5) / 3)
    area_mommi_expr = (
        2 * np.ceil(np.log2(P)) * Y + P * MMI + (P - 1) * (2 * Y + PS) + 2 * np.ceil(np.log2(P)) * (k - 1) * CR
    )
    area_mzi = (2 * k + 1) * (2 * DC + 2 * PS)
    k0 = 4
    area_osnn4 = (
        2 * np.ceil(np.log2(k / k0)) * Y
        + (2 + 2 * np.ceil(np.log2(k0))) * (DC + PS)
        + (2 * np.ceil(np.log2(k / k0)) * (k0 - 1) + 2) * CR
    )
    k0 = 8
    area_osnn8 = (
        2 * np.ceil(np.log2(k / k0)) * Y
        + (2 + 2 * np.ceil(np.log2(k0))) * (DC + PS)
        + (2 * np.ceil(np.log2(k / k0)) * (k0 - 1) + 8) * CR
    )

    for i in area_mommi_log:
        print(i)
    print()
    for i in area_mommi_expr:
        print(i)
    print()
    for i in area_mzi:
        print(i)
    print()
    for i in area_osnn4:
        print(i)
    print()
    for i in area_osnn8:
        print(i)


def plot_delay_compare():
    OEEO = 10 + 10 + 200  # ps
    MMI = 55.4
    PS = 90
    Y = 1.8
    DC = 29.3
    CR = 7.4
    k = np.arange(4, 65)
    logk = np.floor(np.log2(k))
    n_g = 4.3
    c = 299792458  # m / s
    MMI = MMI * k / 4
    area_mommi_log = OEEO + (2 * Y + logk * MMI + (logk - 1) * (2 * Y + PS) + 2 * (k - 1) * CR) * n_g / c * 1e6
    alpha = 0.7
    P = np.round((1 + (1 + 6 * alpha * k) ** 0.5) / 3)
    C = np.ceil((1 + (1 + 6 * alpha * k) ** 0.5) / 3)
    area_mommi_expr = (
        OEEO
        + (2 * np.ceil(np.log2(P)) * Y + C * MMI + (C - 1) * (2 * Y + PS) + 2 * np.ceil(np.log2(P)) * (k - 1) * CR)
        * n_g
        / c
        * 1e6
    )
    area_mzi = OEEO + ((2 * k + 1) * (2 * DC + 2 * PS)) * n_g / c * 1e6

    k0 = 4
    area_osnn4 = (
        OEEO
        + (
            2 * np.ceil(np.log2(k / k0)) * Y
            + (2 + 2 * np.ceil(np.log2(k0))) * (DC + PS)
            + (2 * np.ceil(np.log2(k / k0)) * (k0 - 1) + 2) * CR
        )
        * n_g
        / c
        * 1e6
    )
    k0 = 8
    area_osnn8 = (
        OEEO
        + (
            2 * np.ceil(np.log2(k / k0)) * Y
            + (2 + 2 * np.ceil(np.log2(k0))) * (DC + PS)
            + (2 * np.ceil(np.log2(k / k0)) * (k0 - 1) + 8) * CR
        )
        * n_g
        / c
        * 1e6
    )

    for i in area_mommi_log:
        print(i)
    print()
    for i in area_mommi_expr:
        print(i)
    print()
    for i in area_mzi:
        print(i)
    print()
    for i in area_osnn4:
        print(i)
    print()
    for i in area_osnn8:
        print(i)


if __name__ == "__main__":
    # plot_matrices()
    # plot_area_compare()
    # plot_IL_compare()
    plot_delay_compare()
