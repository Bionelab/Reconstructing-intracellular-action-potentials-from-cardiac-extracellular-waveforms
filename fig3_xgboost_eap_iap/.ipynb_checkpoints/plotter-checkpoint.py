import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from colorsys import rgb_to_hls, hls_to_rgb


def adjust_lightness(color, amount=1.0):
    """ Adjust the lightness of a given color """
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    r, g, b = mcolors.to_rgb(c)
    h, l, s = rgb_to_hls(r, g, b)
    l = max(min(l * amount, 1.0), 0.0)
    return hls_to_rgb(h, l, s)


def all_plotters(data, iap = True):
        if iap:
            base_color_hex = '#6DC8BF'
            lw = 0.1
            iap = data.intras_raw
        else:
            base_color_hex = '#F68B1F'
            lw = 0.01
            iap = data.extras_raw
        base_color_rgb = mcolors.hex2color(base_color_hex)

        num_colors = len(iap)  # Assuming 243 samples as per your data
        colors = [adjust_lightness(base_color_rgb, amount=1.0 - 0.5 * (i / num_colors)) for i in range(num_colors)]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=num_colors)
        fig, ax = plt.subplots(figsize=(10, 6))
        norm = plt.Normalize(0, iap.shape[0])
        for idx, sample in enumerate(iap):
            ax.plot(sample- sample [0], color=cmap(norm(idx)), linewidth=0.1, rasterized=True)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, aspect=5)
        cbar.set_label('Sample Number')
        tick_locs = np.arange(0, iap.shape[0], iap.shape[0] // 10)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(tick_locs)
        plt.savefig('iaps.svg', format='svg')
        plt.show()


def corr_heatmap(df,eap_columns,iap_columns, name):

        corr_matrix = np.empty((len(eap_columns), len(iap_columns)))
        for i, eap_col in enumerate(eap_columns):
            for j, iap_col in enumerate(iap_columns):
                corr_matrix[i, j] = df[eap_col].corr(df[iap_col])

        corr_df = pd.DataFrame(corr_matrix, index=eap_columns, columns=iap_columns)
        plt.figure(figsize=(10,8), dpi = 250)
        sns.set(style="white")  # Setting the style for aesthetics
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8, "color": "black"})
        plt.title('Correlation Heatmap between EAP and IAP columns')
        plt.xlabel('IAP Columns')
        plt.ylabel('EAP Columns')
        plt.savefig('results'+name+'correlation_heatmap.svg', format='svg')
        plt.show()


