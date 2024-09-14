import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter
from colorsys import rgb_to_hls, hls_to_rgb
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from xg_utils import *

def model_and_errors(best_params,x_train,y_train,x_val,y_val,x_test1,y_test1,x_test2,y_test2,x_test3,y_test3):

    best_model = XGBRegressor(**best_params, early_stopping_rounds=10)
    best_model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    
    # Predictions
    preds_train = best_model.predict(x_train)
    preds_test1 = best_model.predict(x_test1)
    preds_test2 = best_model.predict(x_test2)
    preds_test3 = best_model.predict(x_test3)
    
    # Calculate errors
    error_train_perc = np.abs(preds_train - y_train)*100/y_train
    error_test1_perc = np.abs(preds_test1 - y_test1)*100/y_test1
    error_test2_perc = np.abs(preds_test2 - y_test2)*100/y_test2
    error_test3_perc = np.abs(preds_test3 - y_test3)*100/y_test3
    
    error_train = np.abs(preds_train - y_train)
    error_test1 = np.abs(preds_test1 - y_test1)
    error_test2 = np.abs(preds_test2 - y_test2)
    error_test3 = np.abs(preds_test3 - y_test3)
    return best_model, [preds_train,preds_test1,preds_test2,preds_test3],[error_train,error_test1,error_test2,error_test3],[error_train_perc,error_test1_perc,error_test2_perc,error_test3_perc]
    
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



def plot_test_comparison(y_tests, preds_tests, titles_mae=['First Signal', 'Last Signal']):
    """
    Function to plot comparisons between actual and predicted test data signals.
    
    Parameters:
    - y_tests: List of actual test data (y_test1, y_test2, y_test3).
    - preds_tests: List of predicted test data (preds_test1, preds_test2, preds_test3).
    - titles_mae: Titles for the plots (default is ['First Signal', 'Last Signal']).
    """
    # Loop over each test data and its corresponding predictions
    for test_num, (y_test, preds) in enumerate(zip(y_tests, preds_tests), start=1):
        # Convert predictions to DataFrame (if necessary)
        preds_df = pd.DataFrame(preds, index=y_test.index, columns=y_test.columns)

        # Calculate errors
        error = abs(preds_df - y_test)

        # Sorting and selecting specific indices based on the error
        c = error.sum(axis=1).sort_values()
        locs = [sorted(c.index)[5], sorted(c.index)[-5]]  # First and last signals

        # Plot setup
        fig = plt.figure(figsize=(10*2/3, 8/3))
        gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.1, wspace=0.1)

        # Loop over the selected sample locations (first and last)
        for i, sample_id in enumerate(locs):
            ax = fig.add_subplot(gs[0, i])  # Add a subplot in the grid
            values = np.array(y_test.loc[sample_id])  # Actual values
            value2 = np.array(preds_df.loc[sample_id])  # Predicted values

            # Plot the lines for each sample
            for k, (value, value2) in enumerate(zip(reversed(values), reversed(value2))):
                ax.hlines(k, 0.2, 0.2 + value, colors='blue')  # Actual values in blue
                ax.hlines(k - 0.5, 0.2, 0.2 + value2, colors='red')  # Predicted values in red

            ax.label_outer()  # Only show outer tick labels
            ax.set_xlim(0, 1.6)
            ax.set_ylim(-2, 10)
            ax.set_title(titles_mae[i], fontsize=10)

        # Show the plot
        plt.show()

def apd_dis_errors(abs_apd_errors,perc_apd_errors):
    error_train,error_test1,error_test2,error_test3 = abs_apd_errors
    error_train_perc,error_test1_perc,error_test2_perc,error_test3_perc = perc_apd_errors
    df_train2 = pd.DataFrame()
    df_train2[["APD " + str(10 * i) for i in range(1, 11)]] = error_train
    
    df_test1_2 = pd.DataFrame()
    df_test1_2[["APD " + str(10 * i) for i in range(1, 11)]] = error_test1
    
    df_test2_2 = pd.DataFrame()
    df_test2_2[["APD " + str(10 * i) for i in range(1, 11)]] = error_test2
    
    df_test3_2 = pd.DataFrame()
    df_test3_2[["APD " + str(10 * i) for i in range(1, 11)]] = error_test3
    
    # Define colors suitable for high-quality publications
    color_train = "#1f77b4"  # Muted blue
    color_test1 = "#ff7f0e"  # Muted orange
    color_test2 = "#2ca02c"  # Muted green
    color_test3 = "#d62728"  # Muted red (for the third test)
    
    colors_text = [color_train, color_test1, color_test2, color_test3]  # Solid colors for text
    fig, ax = plt.subplots(figsize=(16, 5), dpi=250)
    
    # Plotting
    for i, column in enumerate(df_train2.columns):
        for df_, color, offset, label in zip([df_train2, df_test1_2, df_test2_2, df_test3_2],
                                             [color_train, color_test1, color_test2, color_test3],
                                             [-0.3, -0.1, 0.1, 0.3],
                                             ['Train', 'Test 1', 'Test 2', 'Test 3']):
            parts = ax.violinplot(df_[column], positions=[i + offset], showmeans=False, showmedians=False, showextrema=False, widths=0.2)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
    
            bp = ax.boxplot(df_[column], positions=[i + offset], notch=True, patch_artist=True, zorder=10, widths=0.2, showfliers=False)
            for patch in bp['boxes']:
                patch.set_facecolor('white')
                patch.set_edgecolor('black')
            for whisker in bp['whiskers']:
                whisker.set(color='black', linestyle='-')
            for cap in bp['caps']:
                cap.set(color='black', linestyle='-')
            for median in bp['medians']:
                median.set(color='black', linestyle='-')
    
    # Aesthetic adjustments
    ax.set_xticks(np.arange(len(df_train2.columns)))
    ax.set_xticklabels(df_train2.columns, rotation=0, ha='center', fontsize=6)
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')
    
    ax.set_ylabel('APD Error (s)', fontsize=14, color='black')
    ax.set_title("Distribution of Errors in APDs Predicted by XGBoost", fontsize=14, color='black')
    
    ax.tick_params(axis='y', labelsize=6)
    
    # Adjust legend order: Train, Test 1, Test 2, Test 3
    legend_elements = [
        Patch(facecolor=color_train, edgecolor='black', alpha=0.7, label='Train'),
        Patch(facecolor=color_test1, edgecolor='black', alpha=0.7, label='Test 1'),
        Patch(facecolor=color_test2, edgecolor='black', alpha=0.7, label='Test 2'),
        Patch(facecolor=color_test3, edgecolor='black', alpha=0.7, label='Test 3')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=6)
    
    plt.tight_layout()
    # plt.savefig("results/xg_train_test1_test2_test3.svg", format="svg", dpi=350)
    plt.show()


    # Assuming sum_error_train, sum_error_test1, sum_error_test2, and sum_error_test3 are already calculated
    sum_error_test3 = np.abs(error_test3).mean(axis=1)
    sum_error_test2 = np.abs(error_test2).mean(axis=1)
    sum_error_test1 = np.abs(error_test1).mean(axis=1)
    sum_error_train = np.abs(error_train).mean(axis=1)
    
    # Add colors for each dataset
    color_train = "#1f77b4"  # Muted blue for Train
    color_test1 = "#ff7f0e"  # Muted orange for Test 1
    color_test2 = "#2ca02c"  # Muted green for Test 2
    color_test3 = "#d62728"  # Muted red for Test 3
    
    colors_text = [color_train, color_test1, color_test2, color_test3]  # Colors for the text annotations
    
    fig, ax = plt.subplots(figsize=(4, 4), dpi=250)
    
    for counter, (data, color_text) in enumerate(zip([sum_error_train, sum_error_test1, sum_error_test2, sum_error_test3], colors_text)):
        parts = ax.violinplot(data, positions=[counter], showmeans=False, showmedians=False, showextrema=False, widths=0.3)
        for pc in parts['bodies']:
            pc.set_facecolor(color_text)  # Use the correct color for each dataset
            pc.set_alpha(0.7)
    
        bp = ax.boxplot(data, positions=[counter], notch=True, patch_artist=True, zorder=10, widths=0.3, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_edgecolor('black')
        for whisker in bp['whiskers']:
            whisker.set(color='black', linestyle='-')
        for cap in bp['caps']:
            cap.set(color='black', linestyle='-')
        for median in bp['medians']:
            median.set(color='black', linestyle='-')
    
        # Add text annotation for median value and standard deviation
        median_value = np.mean(data)
        std_ = np.std(data)
        ax.text(counter, np.max(data) * 1.01, f'{median_value:.2f} ± {std_:.2f}', ha='center', va='bottom', color=color_text, fontsize=6)
    
    # Aesthetic adjustments
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Train', 'Test 1', 'Test 2', 'Test 3'], rotation=0, ha='center', fontsize=8)
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.yaxis.label.set_color('black')
    ax.xaxis.label.set_color('black')
    ax.title.set_color('black')
    
    ax.set_ylabel('Mean APD Error (s)', fontsize=10, color='black')
    ax.set_title("Mean APD Errors in Train, Test 1, Test 2, and Test 3 Sets", fontsize=5, color='black')
    
    plt.tight_layout()
    # plt.savefig("results/xg_sum_apd_error_with_test3.svg", format="svg", dpi=350)
    plt.show()
    
    # Print mean, standard deviation, and maximum error for each dataset
    print("Train Mean:", sum_error_train.mean(), "Std:", sum_error_train.std(), "Max:", sum_error_train.max())
    print("Test 1 Mean:", sum_error_test1.mean(), "Std:", sum_error_test1.std(), "Max:", sum_error_test1.max())
    print("Test 2 Mean:", sum_error_test2.mean(), "Std:", sum_error_test2.std(), "Max:", sum_error_test2.max())
    print("Test 3 Mean:", sum_error_test3.mean(), "Std:", sum_error_test3.std(), "Max:", sum_error_test3.max())


    for counter, test_ in enumerate([error_test1_perc,error_test2_perc,error_test3_perc]):
        mean_errors = test_.mean(axis=0)
        std_errors = test_.std(axis=0)
        apds_of_interest = ["APD30", "APD50", "APD70", "APD90"]
        for apd in apds_of_interest:
            mean = mean_errors[apd]
            std = std_errors[apd]
            print(f"TEST {counter+1}__{apd}: {mean:.3f} ± {std:.3f}")



def plot_distribution(df_xg2, test_names, all_eap_cols, iap_columns):
    """
    Function to create violin plots with overlaid boxplots for Train and Test sets.
    
    Parameters:
    - df_xg2: The input DataFrame containing all data.
    - test_names: A list of test 'name' values (e.g., Test1, Test2, Test3).
    - all_eap_cols: List of column names for EAP-related data.
    - iap_columns: List of column names for IAP-related data.
    """
    
    # Reset the index of the DataFrame
    df_xg2 = df_xg2.reset_index()
    
    # Create a DataFrame for Test sets based on the provided names
    df_tests = [df_xg2[df_xg2['name'] == test_name] for test_name in test_names]
    
    # Any name not in test_names is considered as part of the training set
    df_train = df_xg2[~df_xg2['name'].isin(test_names)]
    
    # Add 'Set' column to each DataFrame
    df_train['Set'] = 'Train'
    
    # Assign 'Set' labels for test sets dynamically
    for i, df_test in enumerate(df_tests):
        df_test['Set'] = f'Test{i+1}'
    
    # Merge all DataFrames (Train and all Test sets)
    df_merged = pd.concat([df_train] + df_tests)

    # Define grid size for plots
    n_cols = 5
    n_rows = len(all_eap_cols + iap_columns) // n_cols + (len(all_eap_cols + iap_columns) % n_cols > 0)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 2, n_rows * 3), constrained_layout=True, dpi=250)
    axes = axes.flatten()

    # Define color palette for sets
    palette = {'Train': 'skyblue', 'Test1': 'salmon', 'Test2': 'lightgreen', 'Test3': 'purple'}
    
    # Loop through all columns and create violin plots with box plots
    for i, var in enumerate(all_eap_cols + iap_columns):
        ax = axes[i]
        sns.violinplot(x='Set', y=var, data=df_merged, ax=ax, palette=palette, inner=None)
        
        # Create box plots for each Set (Train, Test1, Test2, etc.)
        for j, label in enumerate(['Train'] + [f'Test{i+1}' for i in range(len(test_names))]):
            df = df_merged[df_merged['Set'] == label]
            box_data = df[var]
            ax.boxplot(box_data, positions=[j], widths=0.5, patch_artist=True,
                       notch=True,
                       boxprops=dict(facecolor='white', color='black', alpha=0.9),
                       capprops=dict(color='black', alpha=0.9),
                       whiskerprops=dict(color='black', alpha=0.9),
                       flierprops=dict(color='black', markeredgecolor='black', alpha=0.5),
                       medianprops=dict(color='black'),
                       showfliers=False)
        ax.set_title(var, fontsize=14, y=1.09)
        ax.set_xticks(range(len(['Train'] + [f'Test{i+1}' for i in range(len(test_names))])))
        ax.set_xticklabels(['Train'] + [f'Test{i+1}' for i in range(len(test_names))])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Turn off unused axes
    for ax in axes[len(all_eap_cols + iap_columns):]:
        ax.axis('off')

    # Create legend dynamically
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[label]) for label in ['Train'] + [f'Test{i+1}' for i in range(len(test_names))]]
    labels = ['Train'] + [f'Test{i+1}' for i in range(len(test_names))]
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 0.95))

    # plt.savefig('results/distribution.svg', format='svg')
    plt.show()
