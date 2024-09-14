import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from utils import *
from ap import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import load_model
from pia_unet import *
from constants import *
def act_vs_pred_plot(x_values, y_values, label, color, plot_name, save_path):
    """
    Creates a scatter plot of actual vs. predicted values with a reference line and correlation value.
    
    Parameters:
    - x_values: Actual values
    - y_values: Predicted values
    - label: Label for the scatter plot (used in the legend)
    - color: Color for the scatter plot points
    - plot_name: Name of the plot (used as the title)
    - save_path: File path to save the plot
    """
    # Calculate correlation coefficient
    correlation = np.corrcoef(x_values.flatten(), y_values.flatten())[0, 1]

    x_values = x_values[:,100:-100]
    y_values = y_values[:,100:-100]
    plt.figure(figsize=(5, 5))  # Create a new figure
    scatter = plt.scatter(x_values, y_values, s=0.1, alpha=0.05, color=color, label=label)
    scatter.set_rasterized(True)  # Rasterize the scatter plot

    # Set limits
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)

    # Plot a reference line (y = x) to show ideal predictions
    plt.plot([-0.2, 1.2], [-0.2, 1.2], color='gray', linestyle='--', linewidth=1)

    # Add correlation text to the plot (inside the plot area)
    plt.text(0.1, 0.9, f'Correlation = {correlation:.2f}', fontsize=12, color='black', 
             ha='left', va='center', transform=plt.gca().transAxes)

    # Add labels and title
    plt.title(f'{plot_name}')  # Title without correlation
    plt.xlabel('Actual Values')  # Label for the x-axis
    plt.ylabel('Predicted Values')  # Label for the y-axis
    plt.legend()  # Add legend to the plot
    
    # Save the plot as SVG
    # plt.savefig(save_path + '/' + plot_name + '.svg', format="svg", dpi=350)
    plt.show()  # Display the plot




def create_overlay_plot(models_list, tests_list, data, xxx, save_dir=None):
    """
    Creates an overlay plot of multiple test sets with different colors.
    
    Parameters:
    - models_list: List of models to load and evaluate
    - tests_list: List of test sets to use
    - data: Dictionary containing the test data
    - name_to_dic: Function to extract model metadata
    - save_dir: Directory to save the plot (optional)
    """
    plt.figure(figsize=(10, 10))

    colors = ['blue', 'green', 'orange']  # Colors for different test sets
    for model_number, selected_model in enumerate(models_list):
        print(selected_model)
        model = load_model(selected_model, compile=False)
        
        for test_number, test_data in enumerate(tests_list):
            print(test_data)
            extras_test_ = data[(test_data, 'extra')]
            intras_test_ = data[(test_data, 'intra')]
            dic = name_to_dic(selected_model)
            
            if dic['physics'] == 'True':
                preds = model.predict(extras_test_)[0].reshape(-1, 8000)  # Adjusted based on your prediction output structure
                
                act_vs_pred_plot(preds.flatten(), intras_test_.flatten(), label=f'Test {test_number + 1}', color=colors[test_number])

    # Plot the x=y line
    line_range = [min(min(preds.flatten()), min(intras_test_.flatten())), max(max(preds.flatten()), max(intras_test_.flatten()))]
    plt.plot(line_range, line_range, color='red', linestyle='--', label='x = y')

    plt.title('Overlay of Actual vs. Predicted Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    # plt.grid(True)
    # plt.legend()
    
    # if save_dir:
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # save_path = os.path.join(save_dir, "act_vs_preds_overlay.svg")
        # plt.savefig(save_path+xxx+'.svg', format="svg", dpi=250,)
        # print(f"Plot saved to {save_path}")
    
    plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def act_vs_pred_plot_svg(x_values, y_values, label, color, plot_name, save_path):
    """
    Creates a scatter plot of actual vs. predicted values with a reference line and R² value, and saves it as an SVG file.
    
    Parameters:
    - x_values: Actual values
    - y_values: Predicted values
    - label: Label for the scatter plot (used in the legend)
    - color: Color for the scatter plot points
    - plot_name: Name of the plot (used as the title)
    - save_path: File path to save the plot
    """
    # Calculate R² score
    r2 = r2_score(x_values, y_values)
    correlation_matrix = np.corrcoef(x_values, y_values)
    correlation = correlation_matrix[0, 1]


    plt.figure(figsize=(5, 5))  # Create a new figure
    scatter = plt.scatter(x_values, y_values, s=0.1, alpha=0.05, color=color, label=label)

    # Set limits
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)

    # Plot a reference line (y = x) to show ideal predictions
    plt.plot([-0.2, 1.2], [-0.2, 1.2], color='gray', linestyle='--', linewidth=1)

    # Add R² text to the plot
    plt.text(0.05, 1.05, f'R² = {r2:.2f}', fontsize=12, color='black', 
             ha='left', va='center', transform=plt.gca().transAxes)

    # Add labels and title
    plt.title(f'{plot_name}')  # Title without R²
    plt.xlabel('Actual Values')  # Label for the x-axis
    plt.ylabel('Predicted Values')  # Label for the y-axis
    plt.legend()  # Add legend to the plot

    # # Save the plot as SVG (fully vectorized, no rasterization)
    # svg_path = f"{save_path}/{plot_name}.svg"
    # plt.savefig(svg_path, format="svg", dpi=350)
    # print(f"Plot saved to {svg_path}")

    plt.show()  # Display the plot

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp

def get_sample_indices(data_length, total_samples, seed=42):
    """
    Generates sample indices evenly spaced across the sorted data.
    
    Parameters:
    - data_length: The total length of the data.
    - total_samples: The total number of samples to be drawn.
    - seed: An integer to seed the random number generator for repeatability.
    
    Returns:
    - indices: A list of selected indices.
    """
    if seed is not None:
        np.random.seed(seed)
    
    step_size = max(1, data_length // total_samples)
    indices = np.arange(0, data_length, step_size)[:total_samples]
    
    return indices

def plot_samples(name, intras_normalized, preds, total_samples=10, save_dir=None):
    """
    Plots a specified number of intras_normalized and preds data.
    
    Parameters:
    - name: Name of the plot.
    - intras_normalized: The normalized data to plot.
    - preds: The predicted data to plot.
    - total_samples: Total number of samples to plot.
    - save_dir: Directory to save the plot.
    """
    intras_normalized = np.array(intras_normalized)[5:-10]
    preds = np.array(preds)[5:-10]
    
    data_length = min(len(intras_normalized), len(preds))
    sample_indices = get_sample_indices(data_length, total_samples)

    # Find global y-axis limits
    y_min = -0.2
    y_max = 1.2
    
    fig, axes = plt.subplots(nrows=1, ncols=total_samples, figsize=(4 * total_samples, 4))
    for i, idx in enumerate(sample_indices):
        axes[i].plot(np.arange(len(intras_normalized[idx])), intras_normalized[idx], color='black')
        axes[i].plot(np.arange(len(preds[idx])), preds[idx], color='red')
        axes[i].set_title(f'Sample {idx}')
        axes[i].set_ylim([y_min, y_max])
    
    plt.tight_layout()
    
    # if save_dir:
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    #     save_path = os.path.join(save_dir, f'{name}.svg')
    #     plt.savefig(save_path, format='svg', bbox_inches='tight')
    #     print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_physics(ax, intras_test_, v_out, dv_out_max, dv_t, dv2_t, 
                 a_oot, k_oot, x_oot, t_oot, dv_l, dv_r, t_out_broad):
    """
    Plots physics-related data on the given axis.
    """
    v_out = v_out.reshape(-1, 8000)
    dv_out_max = dv_out_max.reshape(dv_out_max.shape[0], -1)
    dv_t = dv_t.reshape(dv_t.shape[0], -1)
    dv2_t = dv2_t.reshape(dv2_t.shape[0], -1)
    a_oot = a_oot.reshape(a_oot.shape[0], -1)
    k_oot = k_oot.reshape(k_oot.shape[0], -1)
    x_oot = x_oot.reshape(x_oot.shape[0], -1)
    t_oot = t_oot.reshape(t_oot.shape[0], -1)

    for i in range(len(dv_out_max)):  # Loop through dv_out_max
        v = v_out[i]
        dvt = dv_t[i]
        dvtt = dv2_t[i]
        a = a_oot[i][0]
        k = k_oot[i][0]
        x = x_oot[i][0]
        t = t_oot[i][0]
        
        # Evaluate the ODE with the given parameters
        t_eval = np.linspace(0, t, 8000)
        t_span = (0, t)
        solution_opt = solve_ivp(aliev_panfilov_ode, t_span, [0.1, 0.4], 
                                 args=(a, k, x), t_eval=t_eval, method='RK45')
        ap_rec = solution_opt.y[0]
        
        # Plot the data on the provided axis
        ax.plot(intras_test_[i], label='intra', color='black', alpha=0.1)
        ax.plot(v, label='pred', color='red', alpha=0.1)
        ax.plot(ap_rec / np.max(ap_rec), label='ap', color='green', alpha=0.3)
    
    # Set title and legend for the axis (if desired)
    # ax.set_title("Physics Dat
def apd_comp_plot(apds_pred,apds_test,apds_pred_train,apds_train,name):
        error =  apds_pred - apds_test
        error_train = (apds_pred_train- apds_train)
        df = np.abs(error)/5000
        df2 = pd.DataFrame()
        df2[["APD "+str(10*i) for i in range(1,11)] ]= df
        
        df_train =np.abs(error_train)/5000
        df_train2 = pd.DataFrame()
        df_train2[["APD "+str(10*i) for i in range(1,11)] ]= df_train
        
        colors_green= LinearSegmentedColormap.from_list("MyCmapNameBlue", ["#000080", "#000080"])
        colors_purple = LinearSegmentedColormap.from_list("MyCmapNameOrange", ["#FF4500","#FF4500"])
        
        colors_text = ["#000080", "#FF4500"]  # Solid colors for text
        fig, ax = plt.subplots(figsize=(5, 3), dpi=250)
        
        for i, column in enumerate(df2.columns):
            c = 0
            for df_, colors, offset, label, color_text in zip([ df2,df_train2], [colors_green, colors_purple], [-0.2, 0.2], ['Test', 'Train'], colors_text):
                parts = ax.violinplot(df_[column], positions=[i+offset], showmeans=False, showmedians=False, showextrema=False, widths=0.3)
                for pc in parts['bodies']:
                    pc.set_facecolor(colors(i/10))
                    # pc.set_edgecolor('black')
                    pc.set_alpha(0.7)
        
                bp = ax.boxplot(df_[column], positions=[i+offset], notch=True, patch_artist=True, zorder=10, widths=0.3, showfliers=False)
                for patch in bp['boxes']:
                    patch.set_facecolor('white') 
                    patch.set_edgecolor('black')
                for whisker in bp['whiskers']:
                    whisker.set(color='black', linestyle='-')
                for cap in bp['caps']:
                    cap.set(color='black', linestyle='-')
                for median in bp['medians']:
                    median.set(color='black', linestyle='-')
                median_value = np.mean(df_[column])
                std_ = np.std(df_[column])
        
                c = c+1
        ax.set_xticks(np.arange(len(df2.columns)))
        xticks = ax.get_xticklabels()
        for tick in xticks:
            tick.set_color('black')
        ax.set_xticklabels(df2.columns, rotation=0, ha='center', fontsize=4)
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.yaxis.label.set_color('black')
        ax.xaxis.label.set_color('black')
        ax.title.set_color('black')
        
        ax.set_ylabel('APD Error (s)', fontsize=14, color='black')
        ax.set_title("Distribution of Errors in APDs Predicted by XGBoost", fontsize=14, color='black')
        
        ax.tick_params(axis='y', labelsize=4)
        
        legend_elements = [
                           Patch(facecolor=colors_purple(0.5), edgecolor='black', alpha=0.7, label='Test'),
                          Patch(facecolor=colors_green(0.5), edgecolor='black', alpha=0.7, label='Train')]
        # ax.legend(handles=legend_elements, loc='upper left', fontsize = 16)
        ax.legend(handles=legend_elements, loc='upper left', fontsize = 4)
        
        plt.tight_layout()
        # plt.savefig("models_to_comp/"+name+"apds_test_train.svg", format="svg",  dpi = 350)
        plt.show()


def model_plotter_apd(preds_,extras_test2 ,intras_test2,name_, ph ):

    preds = preds_.copy()
    print(preds.shape)
    color = sns.color_palette("Set1")
    # Arrays for the titles of each column
    mid0 = round(len(extras_test2)*0.25)
    mid1 = round(len(extras_test2)*0.5)
    mid2 = round(len(extras_test2)*0.75)
    preds_mae = preds[[1,-2]]
    intras = intras_test2[[1,-2]]
    extras = extras_test2[[1,-2]]
    n_cols = len([1,-2])
    print(n_cols)
    time = np.arange(1, 1601, 0.2)
    fig = plt.figure(figsize=((12/5)*n_cols, 4), dpi=250)
    color = sns.color_palette("Set1")
    gs = gridspec.GridSpec(2, n_cols, figure=fig)  # Create 3x6 grid
    for i in range(n_cols):
        print(preds_.shape)

        ax = fig.add_subplot(gs[0, i])  # Add a subplot in the current grid position
        ax.label_outer()  # Only show outer tick labels
        ax.plot(time,preds_mae[i], label='Actual', linestyle='-', color=color[1],linewidth=3)
        ax.plot(time,  intras[i], label='Predicted', linestyle='-.', color=color[0],linewidth=3)

        ax = fig.add_subplot(gs[1, i])  # Add a subplot in the current grid position
        ax.label_outer()  # Only show outer tick labels
        ax.plot(time,  extras[i], label='Predicted', linestyle='-.', color='black',linewidth=1)

    plt.tight_layout()
    # plt.savefig("models_to_comp/"+name_+"_apd_range_preds.svg", format="svg",  dpi = 350)
    plt.show()

def plot_apds(loc, model_name, actual_apds_list, apd_preds_list):
    apd_train, apd_test1, apd_test2, apd_test3 = actual_apds_list
    apd_p_train, apd_p_test1, apd_p_test2, apd_p_test3 = apd_preds_list

    error_train_perc = np.abs(apd_p_train - apd_train) * 100 / apd_train
    error_test1_perc = np.abs(apd_p_test1 - apd_test1) * 100 / apd_test1
    error_test2_perc = np.abs(apd_p_test2 - apd_test2) * 100 / apd_test2
    error_test3_perc = np.abs(apd_p_test3 - apd_test3) * 100 / apd_test3

    error_train = np.abs(apd_p_train - apd_train) / 5000
    error_test1 = np.abs(apd_p_test1 - apd_test1) / 5000
    error_test2 = np.abs(apd_p_test2 - apd_test2) / 5000
    error_test3 = np.abs(apd_p_test3 - apd_test3) / 5000

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

    fig, ax = plt.subplots(figsize=(16, 5), dpi=250)

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
    ax.set_ylabel('APD Error (s)', fontsize=14, color='black')
    ax.set_title(f"Distribution of Errors in APDs Predicted by {model_name}", fontsize=14, color='black')

    ax.tick_params(axis='y', labelsize=6)

    legend_elements = [
        Patch(facecolor=color_train, edgecolor='black', alpha=0.7, label='Train'),
        Patch(facecolor=color_test1, edgecolor='black', alpha=0.7, label='Test 1'),
        Patch(facecolor=color_test2, edgecolor='black', alpha=0.7, label='Test 2'),
        Patch(facecolor=color_test3, edgecolor='black', alpha=0.7, label='Test 3')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=6)
    plt.ylim(0, 0.3)
    plt.tight_layout()

    # Save the figure with loc and model name in the filename
    # plt.savefig(f"{loc}/{model_name}_apd_range_preds.svg", format="svg", dpi=350)
    plt.show()

    # Assuming sum_error_train, sum_error_test1, sum_error_test2, and sum_error_test3 are already calculated
    sum_error_train = np.abs(error_train).mean(axis=1)
    sum_error_test1 = np.abs(error_test1).mean(axis=1)
    sum_error_test2 = np.abs(error_test2).mean(axis=1)
    sum_error_test3 = np.abs(error_test3).mean(axis=1)

    sum_error_train_perc = np.abs(error_train_perc).mean(axis=1)
    sum_error_test1_perc = np.abs(error_test1_perc).mean(axis=1)
    sum_error_test2_perc = np.abs(error_test2_perc).mean(axis=1)
    sum_error_test3_perc = np.abs(error_test3_perc).mean(axis=1)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=250)

    for counter, (data, color_text) in enumerate(zip([sum_error_train, sum_error_test1, sum_error_test2, sum_error_test3], [color_train, color_test1, color_test2, color_test3])):
        parts = ax.violinplot(data, positions=[counter], showmeans=False, showmedians=False, showextrema=False, widths=0.3)
        for pc in parts['bodies']:
            pc.set_facecolor(color_text)
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
    ax.set_ylabel('Mean APD Error (s)', fontsize=10, color='black')
    ax.set_title(f"Mean APD Errors for {model_name} in Train, Test 1, Test 2, and Test 3", fontsize=5, color='black')

    plt.tight_layout()
    # plt.savefig(f"{loc}/{model_name}_apds_mean.svg", format="svg", dpi=350)
    plt.show()

    # Print statistics
    print(f" Train Mean:", sum_error_train.mean(), "Std:", sum_error_train.std(), "Max:", sum_error_train.max())
    print(f" Test 1 Mean:", sum_error_test1.mean(), "Std:", sum_error_test1.std(), "Max:", sum_error_test1.max())
    print(f" Test 2 Mean:", sum_error_test2.mean(), "Std:", sum_error_test2.std(), "Max:", sum_error_test2.max())
    print(f" Test 3 Mean:", sum_error_test3.mean(), "Std:", sum_error_test3.std(), "Max:", sum_error_test3.max())

    print(f"Train Perc Mean:", sum_error_train_perc.mean(), "Std:", sum_error_train_perc.std(), "Max:", sum_error_train.max())
    print(f"Test Perc 1 Mean:", sum_error_test1_perc.mean(), "Std:", sum_error_test1_perc.std(), "Max:", sum_error_test1.max())
    print(f"Test Perc 2 Mean:", sum_error_test2_perc.mean(), "Std:", sum_error_test2_perc.std(), "Max:", sum_error_test2.max())
    print(f"Test Perc 3 Mean:", sum_error_test3_perc.mean(), "Std:", sum_error_test3_perc.std(), "Max:", sum_error_test3.max())

    return error_train_perc, error_test1_perc, error_test2_perc, error_test3_perc, error_train, error_test1, error_test2, error_test3

def MEA_plot(loc, model_name,preds_list,intras_list):

        # Define colors suitable for high-quality publications
    color_train = "#1f77b4"  # Muted blue
    color_test1 = "#ff7f0e"  # Muted orange
    color_test2 = "#2ca02c"  # Muted green
    color_test3 = "#d62728"  # Muted red (for the third test)

    preds_train,preds1,preds2,preds3 = preds_list
    intra_trian,intra1,intra2,intra3 = intras_list

    sum_error_train = np.mean(np.abs(preds_train-intra_trian), axis = 1)
    sum_error_test1 = np.mean(np.abs(preds1-intra1), axis = 1)
    sum_error_test2 = np.mean(np.abs(preds2-intra2), axis = 1)
    sum_error_test3 = np.mean(np.abs(preds3-intra3), axis = 1)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=250)

    for counter, (data, color_text) in enumerate(zip([sum_error_train, sum_error_test1, sum_error_test2, sum_error_test3], [color_train, color_test1, color_test2, color_test3])):
        parts = ax.violinplot(data, positions=[counter], showmeans=False, showmedians=False, showextrema=False, widths=0.3)
        for pc in parts['bodies']:
            pc.set_facecolor(color_text)
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

    # Print statistics
    print(f" Train Mean:", sum_error_train.mean(), "Std:", sum_error_train.std(), "Max:", sum_error_train.max())
    print(f" Test 1 Mean:", sum_error_test1.mean(), "Std:", sum_error_test1.std(), "Max:", sum_error_test1.max())
    print(f" Test 2 Mean:", sum_error_test2.mean(), "Std:", sum_error_test2.std(), "Max:", sum_error_test2.max())
    print(f" Test 3 Mean:", sum_error_test3.mean(), "Std:", sum_error_test3.std(), "Max:", sum_error_test3.max())

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Train', 'Test 1', 'Test 2', 'Test 3'], rotation=0, ha='center', fontsize=8)
    ax.set_ylabel('MEA', fontsize=10, color='black')
    ax.set_title(f"MEA{model_name} in Train, Test 1, Test 2, and Test 3", fontsize=5, color='black')

    plt.tight_layout()
    # plt.savefig(f"{loc}/{model_name}_MEA.svg", format="svg", dpi=350)
    plt.show()



def run_model_performance(models_list, extras_train, intra_trains, tests, data, actual_apds_list, save_path):
    """
    Trains and tests models, then returns a DataFrame with model performance metrics.

    Parameters:
    - models_list: List of model file paths to be trained and evaluated.
    - extras_train: Input training data (e.g., extracellular data).
    - intra_trains: Ground truth for training data (e.g., intracellular data).
    - tests: List of test data keys.
    - data: Dictionary containing test data for 'extra' and 'intra' channels.
    - actual_apds_list: List of actual APD values.
    - save_path: Directory path to save the results.

    Returns:
    - df: DataFrame summarizing model performance.
    """

    os.makedirs(save_path, exist_ok=True)
    df = pd.DataFrame()

    for model_number, selected_model in enumerate(models_list):
        model_number += 8  # Adjust starting model number if needed
        apds_preds = []
        preds_list = []
        intras_list = []
        dic = name_to_dic(selected_model)
        model = load_model(selected_model, compile=False)

        # Predict on training data
        train_preds = model.predict(extras_train)[0].reshape(-1, 8000)
        apd_train_pred = get_all_apds_multiprocessing(train_preds.reshape(-1, 8000))
        df.loc[model_number, 'MAE_train'] = np.mean(np.abs(train_preds - intra_trains))
        df.loc[model_number, 'MAE_train_std'] = np.std(np.abs(train_preds - intra_trains))
        preds_list.append(train_preds)
        intras_list.append(intra_trains)
        apds_preds.append(apd_train_pred)

        # Create subplots for physics-based model predictions
        fig_physics, axs_physics = plt.subplots(1, len(tests), figsize=(5 * len(tests), 5))

        # Loop over test data
        for test_number, test_data in enumerate(tests):
            extras_test_ = data[(test_data, 'extra')]
            intras_test_ = data[(test_data, 'intra')]

            # Predict and prepare data
            if dic.get('physics') == 'True':
                preds, dv_out_max, dv_t, dv2_t, a_oot, k_oot, x_oot, t_oot, dv_l, dv_r, t_out_broad = model.predict(extras_test_)
                preds = preds.reshape(-1, 8000)
                preds_new, intras_new = autro_correct(preds, intras_test_)
                plot_physics(axs_physics[test_number], intras_new, preds_new, dv_out_max, dv_t, dv2_t, a_oot, k_oot, x_oot, t_oot, dv_l, dv_r, t_out_broad)
                axs_physics[test_number].set_title(f'Test {test_number}')
            else:
                preds = model.predict(extras_test_).reshape(-1, 8000)
                preds_new, intras_new = autro_correct(preds, intras_test_)

            # Calculate MAE and append predictions
            df.loc[model_number, f'MAE_{test_number}'] = np.mean(np.abs(preds_new - intras_new))
            df.loc[model_number, f'MEA-STD_{test_number}'] = np.std(np.abs(preds_new - intras_new))

            for i in dic.keys():
                df.loc[model_number, i] = dic[i]

            preds_list.append(preds_new)
            intras_list.append(intras_new)
            apd_test_pred_ = get_all_apds_multiprocessing(preds_new)
            apds_preds.append(apd_test_pred_)

        model___ = selected_model.split('/')[-1].split('.h5')[0]

        # Save physics plots if applicable
        if dic.get('physics') == 'True':
            plt.tight_layout()
            plt.savefig(f"{save_path}/model_{model___}_physics.png", format='png', dpi=300)
            plt.show()
            plt.close(fig_physics)

        # Plot performance comparison
        MEA_plot(save_path, model___, preds_list, intras_list)
        error_train_perc, error_test1_perc, error_test2_perc, error_test3_perc, error_train, error_test1, error_test2, error_test3 = plot_apds(save_path, model___, actual_apds_list, apds_preds)

        df.loc[model_number,['APD_train_'+str(i) for i in range(1,11)]]=np.mean(error_train, axis = 0)
        df.loc[model_number,['APD_test1_'+str(i) for i in range(1,11)]]=np.mean(error_test1, axis = 0)
        df.loc[model_number,['APD_test2_'+str(i) for i in range(1,11)]]=np.mean(error_test2, axis = 0)
        df.loc[model_number,['APD_test3_'+str(i) for i in range(1,11)]]=np.mean(error_test3, axis = 0)
        
        df.loc[model_number,['APD_train_std'+str(i) for i in range(1,11)]]=np.std(error_train, axis = 0)
        df.loc[model_number,['APD_test1_std'+str(i) for i in range(1,11)]]=np.std(error_test1, axis = 0)
        df.loc[model_number,['APD_test2_std'+str(i) for i in range(1,11)]]=np.std(error_test2, axis = 0)
        df.loc[model_number,['APD_test3_std'+str(i) for i in range(1,11)]]=np.std(error_test3, axis = 0)
        
        df.loc[model_number,['APD_train_perc'+str(i) for i in range(1,11)]]=np.mean(error_train_perc, axis = 0)
        df.loc[model_number,['APD_test1_perc'+str(i) for i in range(1,11)]]=np.mean(error_test1_perc, axis = 0)
        df.loc[model_number,['APD_test2_perc'+str(i) for i in range(1,11)]]=np.mean(error_test2_perc, axis = 0)
        df.loc[model_number,['APD_test3_perc'+str(i) for i in range(1,11)]]=np.mean(error_test3_perc, axis = 0)
        
        df.loc[model_number,['APD_train_perc_std'+str(i) for i in range(1,11)]]=np.mean(error_train_perc, axis = 0)
        df.loc[model_number,['APD_test1_perc_std'+str(i) for i in range(1,11)]]=np.std(error_test1_perc, axis = 0)
        df.loc[model_number,['APD_test2_perc_std'+str(i) for i in range(1,11)]]=np.std(error_test2_perc, axis = 0)
        df.loc[model_number,['APD_test3_perc_std'+str(i) for i in range(1,11)]]=np.std(error_test3_perc, axis = 0)
        
        df.loc[model_number,['APD_train_mean']]=np.mean(np.mean(error_train, axis = 0))
        df.loc[model_number,['APD_test1_mean']]=np.mean(np.mean(error_test1, axis = 0))
        df.loc[model_number,['APD_test2_mean']]=np.mean(np.mean(error_test2, axis = 0))
        df.loc[model_number,['APD_test3_mean']]=np.mean(np.mean(error_test3, axis = 0))
        
        df.loc[model_number,['APD_train_std']]=np.std(error_train)
        df.loc[model_number,['APD_test1_std']]=np.std(error_test1)
        df.loc[model_number,['APD_test2_std']]=np.std(error_test2)
        df.loc[model_number,['APD_test3_std']]=np.std(error_test3)
        
        df.loc[model_number,['APD_train_perc_mean']]=np.mean(np.mean(error_train_perc, axis = 0))
        df.loc[model_number,['APD_test1_perc_mean']]=np.mean(np.mean(error_test1_perc, axis = 0))
        df.loc[model_number,['APD_test2_perc_mean']]=np.mean(np.mean(error_test2_perc, axis = 0))
        df.loc[model_number,['APD_test3_perc_mean']]=np.mean(np.mean(error_test3_perc, axis = 0))
        
        df.loc[model_number,['APD_train_perc_std']]=np.std(error_train_perc)
        df.loc[model_number,['APD_test1_perc_std']]=np.std(error_test1_perc)
        df.loc[model_number,['APD_test2_perc_std']]=np.std(error_test2_perc)
        df.loc[model_number,['APD_test3_perc_std']]=np.std(error_test3_perc)
        df.loc[model_number, 'name'] = selected_model

        # Plot and save samples
        for test_number, test_data in enumerate(tests):
            extras_test_ = data[(test_data, 'extra')]
            intras_test_ = data[(test_data, 'intra')]
            dic = name_to_dic(selected_model)

            if dic.get('physics') == 'True':
                preds = model.predict(extras_test_)[0].reshape(-1, 8000)
            else:
                preds = model.predict(extras_test_).reshape(-1, 8000)
            preds_new, intras_new = autro_correct(preds, intras_test_)

            act_vs_pred_plot(intras_new, preds_new, test_data, 'blue', f'model_{model___}_{test_number}', save_path)
            plot_samples(f"model_{model___}_test_{test_number}_samples", intras_test_, preds, total_samples=15, save_dir=save_path)

    plt.show()  # Show all figures
    return df


