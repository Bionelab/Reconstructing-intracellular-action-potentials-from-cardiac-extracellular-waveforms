import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def box_plot_comparison(df2,columns_to_plot):
# Setting a high-quality, minimalist style
        sns.set(style="white", palette="muted", font_scale=1.2)
        sns.set_context("paper", font_scale=1.5)

        # Creating a figure with adjusted dimensions for compactness
        fig, axes = plt.subplots(1, len(columns_to_plot), figsize=(7.5, 3), dpi=250)

        # Reducing spacing between the plots for a more packed look
        plt.subplots_adjust(wspace=0.3)

        # Creating box plots for each column with jitter for the individual data points
        for i, ax in enumerate(axes):
            ax.grid(True)
            column = columns_to_plot[i]
            sns.boxplot(y=df2[column], ax=ax, color="lightblue", fliersize=0, width=0.6)
            sns.stripplot(y=df2[column], ax=ax, color="black", size=1, jitter=True, alpha=0.6)
            ax.set_title(column)
            ax.set_ylabel('')
            ax.set_xlabel('')
        fig.tight_layout()

        # Saving the plot
        # plt.savefig('box_com_patch_nea.svg', format='svg')

        # Display the plot
        plt.show()


def plot_errors_for_one_sample(df_fig1):
   
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi = 350)
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    al = 0.9
    ax1.scatter(df_fig1['t_pa'], df_fig1['MAEa'], c='blue', marker='s', label='MAE',  s= 5, alpha = al)
    ax1.set_ylabel('MAE', color ='blue')
    ax1.tick_params(axis='y', colors='blue', labelsize=14)
    ax1.set_ylim([0, 0.1])

    ax2.scatter(df_fig1['t_pa'], df_fig1['R2a'], c='red', marker='^', label='r', s= 10, alpha = al)
    ax2.set_ylabel('Correlation', color = 'red')
    ax2.tick_params(axis='y', colors='red', labelsize=14)
    ax2.set_ylim([0.96, 1])

    ax3.scatter(df_fig1['t_pa'], df_fig1['sn_min'], c='orange', marker='*', label='sn_min',  s= 10, alpha = al)
    ax3.spines['right'].set_position(('data', np.max(df_fig1['t_pa'])*1.2))
    ax3.set_ylabel('sn_min', color ='orange')
    ax3.tick_params(axis='y', colors='orange', labelsize=14)
    ax3.spines['left'].set_visible(False)

    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax2.set_ylabel('r', fontsize=12)
    ax3.set_ylabel('sn_min', fontsize=12)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_3, labels_3 = ax3.get_legend_handles_labels()

    ax1.legend(lines_1 + lines_2 +  lines_3, labels_1 + labels_2+  labels_3, loc='upper center', fontsize=14, bbox_to_anchor=(0.5, 1.3), ncol=4)
    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)

   
    plt.xticks(fontsize=12)
    # plt.savefig("figures/patch_errors.svg", format="svg",  dpi = 350)
    plt.show()

    
def plot_cols2(df2, col, col3='sn_a_min',name_='raw', min_group_size=10):
    fig, ax = plt.subplots(figsize=(6, 2), dpi=250)
    group_sizes = df2.groupby(df2.index).size()
    sorted_indices = group_sizes.sort_values(ascending=False).index
    sorted_df2 = pd.concat([df2.loc[idx] for idx in sorted_indices])
    grouped = sorted_df2.groupby(sorted_df2.index)
    sorted_grouped = sorted(grouped, key=lambda x: len(x[1]), reverse=True)
    filtered_grouped = [(name, group) for name, group in sorted_grouped if len(group) >= min_group_size]

    ax2 = ax.twinx()
    median_x = []
    median_y = []
    sem_y = []

    for i, (group_name, group_data) in enumerate(filtered_grouped):
        # Box plot
        bp = ax.boxplot(group_data[col], positions=[i+1], widths=0.8, vert=True, patch_artist=True, showfliers=False)

        for box in bp['boxes']:
            box.set_visible(False)
        for whisker in bp['whiskers']:
            whisker.set_visible(False)
        for cap in bp['caps']:
            cap.set_visible(False)
        for median in bp['medians']:
            median.set(color='red', linewidth=0.7)

        # Jittered scatter over box plot
        jitter = 0.7 * (np.random.rand(len(group_data[col])) - 0.5)
        ax.scatter(np.ones(len(group_data[col])) * (i+1) + jitter, group_data[col], alpha=0.5, color='black', s=0.3)

        # Get median of col3 and append to list
        median_col3 = group_data[col3].median()
        median_x.append(i+1)
        median_y.append(median_col3)

        sem_col3 = group_data[col3].std() / np.sqrt(len(group_data[col3]))
        sem_y.append(sem_col3)

    # Scatter plot for median col3 values with error bars
    ax2.errorbar(median_x, median_y, yerr=sem_y, fmt='o', color='blue', label=f'Median {col3}', markersize=1, marker='.', linestyle='dotted', linewidth=0.3, capsize=2)
    ax2.scatter(median_x, median_y, color='blue', s=1, marker='.')
    ax2.plot(median_x, median_y, color='blue', linestyle='dotted', linewidth=0.3)
    ax2.set_ylabel(col3, color='blue')
    ax2.tick_params(axis='y', colors='blue')

    # Set xticks and labels
    # ax.set_xticks(range(1, len(filtered_grouped) + 1))
    # ax.set_xticklabels([group[0] for group in filtered_grouped], rotation=45, ha='right')
    ax.set_xticks(range(1, len(filtered_grouped) + 1))
    ax.set_xticklabels([f'rec{i+1}' for i in range(len(filtered_grouped))], rotation=45, ha='right')

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(False)
    ax2.grid(False)

    # fig.savefig(name_+col+'_box'+".svg", format='svg')
    plt.show()
    
def apds_overlay(dft2):
        
        s = 2
        width, height = 3, 2
        dpi_val = 350
        fontsize_base = max(width, height) * 2.5  # Adjust this multiplier as needed

        # Create the plot with specified size and DPI
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi_val)
        dft2['start_time'] = dft2['t_pa']
        
        ax.scatter(dft2['start_time'], dft2['APD_patcha5'] / 10000, label='APD50 Patch Clamp', color='#2ca02c', marker='x', s=s)  # darker green
        ax.scatter(dft2['start_time'], dft2['apds_neaa5'] / 10000, label='APD50 NEA', color='#98df8a', marker='.', s=s)  # lighter green


   
        ax.set_ylim(0.1, 0.5)
        # ax.set_xlim(150, 1250)

        # Larger label and legend fonts
        ax.set_xlabel('Time (s)', fontsize=fontsize_base)
        ax.set_ylabel('APD (s)', fontsize=fontsize_base)
        legend = ax.legend(fontsize=fontsize_base - 1, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        # Larger tick labels
        ax.tick_params(axis='both', which='major', labelsize=fontsize_base)
        plt.savefig("patch_iap_apds.svg", format="svg",  dpi = 350)
        plt.show()

def cylce_overlay(dft2):
        
        s = 2
        width, height = 3, 2
        dpi_val = 350
        fontsize_base = max(width, height) * 2.5  # Adjust this multiplier as needed

        # Create the plot with specified size and DPI
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi_val)
        dft2['start_time'] = dft2['t_pa']

        ax.scatter(dft2['start_time'], dft2['dt_pa'], label='Patch Clamp', color='red', marker='x', s=s)  # darker green
        ax.scatter(dft2['start_time'], dft2['dt_na'], label='NEA', color='black', marker='.', s=s)  # lighter green

 
        ax.set_ylim(0.5, 0.7)
        # ax.set_xlim(150, 1250)

        # Larger label and legend fonts
        ax.set_xlabel('Time (s)', fontsize=fontsize_base)
        ax.set_ylabel('APD (s)', fontsize=fontsize_base)
        legend = ax.legend(fontsize=fontsize_base - 1, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(False)

        # Larger tick labels
        ax.tick_params(axis='both', which='major', labelsize=fontsize_base)
        # plt.savefig("patch_iap_apds.svg", format="svg",  dpi = 350)
        plt.show()

        
def plot_average_box(df, columns_of_interest):
    fig, axs = plt.subplots(nrows=1, ncols=len(columns_of_interest), figsize=(10, 2), dpi=250)

    # Custom color for the median line and ticks
    median_color = '#FF5555'  # Light Red

    # Adjust space between the plots
    plt.subplots_adjust(wspace=0.2)

    for i, col in enumerate(columns_of_interest):
        ax = axs[i]

        # Group by index and calculate mean for each group
        group_means = df.groupby(df.index).apply(lambda group: group[col].mean())

        # Jittered scatter with very small, semi-transparent dots
        jitter = 0.15 * (np.random.rand(len(group_means)) - 0.5)
        ax.scatter(np.ones(len(group_means)) + jitter, group_means, alpha=0.8, color='black', s=5)

        # Create minimalist box plot
        bp = ax.boxplot(group_means, vert=True, patch_artist=True, showfliers=False, widths=0.4)
        
        # Hide all elements except median line
        for element in ['boxes', 'whiskers', 'caps']:
            plt.setp(bp[element], visible=False)
            
        # Style median line
        plt.setp(bp['medians'], color=median_color, linewidth=1.2)

        # Set x-tick labels
        ax.set_xticklabels([col])

        # Hide all spines
        # 'left', 'bottom'
        for spine in ['right', 'top', ]:
            ax.spines[spine].set_visible(False)

        # Remove y-axis labels but keep the ticks, set tick color to subtle grey
        ax.yaxis.set_ticks_position('left')
        ax.tick_params(axis='y', colors='black')
        ax.tick_params(axis='x', which='both', labelsize=10, length=0)

    # Save the figure
    fig.savefig('patch_all_errorrs.svg', format='svg')

    plt.show()

def plot_cols2(df2, col, col3='sn_a_min',name_='raw', min_group_size=10):
    fig, ax = plt.subplots(figsize=(6, 2), dpi=250)
    group_sizes = df2.groupby(df2.index).size()
    sorted_indices = group_sizes.sort_values(ascending=False).index
    sorted_df2 = pd.concat([df2.loc[idx] for idx in sorted_indices])
    grouped = sorted_df2.groupby(sorted_df2.index)
    sorted_grouped = sorted(grouped, key=lambda x: len(x[1]), reverse=True)
    filtered_grouped = [(name, group) for name, group in sorted_grouped if len(group) >= min_group_size]

    ax2 = ax.twinx()
    median_x = []
    median_y = []
    sem_y = []

    for i, (group_name, group_data) in enumerate(filtered_grouped):
        # Box plot
        bp = ax.boxplot(group_data[col], positions=[i+1], widths=0.8, vert=True, patch_artist=True, showfliers=False)

        for box in bp['boxes']:
            box.set_visible(False)
        for whisker in bp['whiskers']:
            whisker.set_visible(False)
        for cap in bp['caps']:
            cap.set_visible(False)
        for median in bp['medians']:
            median.set(color='red', linewidth=0.7)

        # Jittered scatter over box plot
        jitter = 0.7 * (np.random.rand(len(group_data[col])) - 0.5)
        ax.scatter(np.ones(len(group_data[col])) * (i+1) + jitter, group_data[col], alpha=0.5, color='black', s=0.3)

        # Get median of col3 and append to list
        median_col3 = group_data[col3].median()
        median_x.append(i+1)
        median_y.append(median_col3)

        sem_col3 = group_data[col3].std() / np.sqrt(len(group_data[col3]))
        sem_y.append(sem_col3)

    # Scatter plot for median col3 values with error bars
    ax2.errorbar(median_x, median_y, yerr=sem_y, fmt='o', color='blue', label=f'Median {col3}', markersize=1, marker='.', linestyle='dotted', linewidth=0.3, capsize=2)
    ax2.scatter(median_x, median_y, color='blue', s=1, marker='.')
    ax2.plot(median_x, median_y, color='blue', linestyle='dotted', linewidth=0.3)
    ax2.set_ylabel(col3, color='blue')
    ax2.tick_params(axis='y', colors='blue')

    # Set xticks and labels
    # ax.set_xticks(range(1, len(filtered_grouped) + 1))
    ax.set_xticklabels([group[0].split('_')[0] for group in filtered_grouped], rotation=45, ha='right')
    # ax.set_xticks(range(1, len(filtered_grouped) + 1))
    # ax.set_xticklabels([f'rec{i+1}' for i in range(len(filtered_grouped))], rotation=45, ha='right')

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(False)
    ax2.grid(False)

    # fig.savefig(name_+col+'_box'+".svg", format='svg')
    plt.show()