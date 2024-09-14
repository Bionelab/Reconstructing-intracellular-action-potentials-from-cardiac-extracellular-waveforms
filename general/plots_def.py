from utils import *

def plotter(iaps,iaps_n,eaps,eaps_n):    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 2x2 grid of subplots
    x = np.arange(8000)  # Assuming all arrays have the same second dimension size

    # Plotting normalized eaps
    ax1 = axs[0]
    ax1.plot(x, iaps.T,  label='eaps')
    ax1.set_ylabel('Normalized eaps', )
    ax1.set_title('Normalized Extra Signals/'   + str(len(eaps)))

    ax1 = axs[1]
    ax1.plot(x, iaps_n.T,  label='iaps')
    ax1.set_ylabel('iaps' )
    ax1.set_title('Intra Signals/'+str(len(iaps)))

    ax1 = axs[2]
    ax1.plot(x, eaps.T,label='eaps')
    ax1.set_ylabel('Normalized eaps',)
    ax1.set_title('Normalized Extra Signals/'  +str(len(eaps_n)))
    ax1 = axs[3]
    ax1.plot(x, eaps_n.T,  label='iaps')
    ax1.set_ylabel('Normalized iaps'  )
    ax1.set_title('Normalized Intra Signals/'+str(len(iaps_n)))

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()



def get_sample_indices(data_length, interval_size, samples_per_interval, seed=42):
    """
    Generates repeatable sample indices within specified intervals.
    
    Parameters:
    - data_length: The total length of the data.
    - interval_size: Fraction of the data length to be considered as an interval.
    - samples_per_interval: The number of samples to be drawn from each interval.
    - seed: An integer to seed the random number generator for repeatability.
    
    Returns:
    - indices: A list of selected indices.
    """
    if seed is not None:
        np.random.seed(seed)
    
    indices = []
    interval_length = int(data_length * interval_size)
    
    for i in range(0, data_length, interval_length):
        interval_indices = np.arange(i, min(i + interval_length, data_length))
        if len(interval_indices) >= samples_per_interval:
            selected_indices = np.random.choice(interval_indices, samples_per_interval, replace=False)
        else:
            selected_indices = interval_indices
        indices.extend(selected_indices)
    
    return indices

def plot_samples(name, intras_normalized, preds, interval_size=0.1, samples_per_interval=10, save_dir=None):
    data_length = len(intras_normalized)
    sample_indices = get_sample_indices(data_length, interval_size, samples_per_interval)
    
    fig, axes = plt.subplots(nrows=1, ncols=samples_per_interval, figsize=(40, 4))
    for i, idx in enumerate(sample_indices[:samples_per_interval]):
        axes[i].plot(np.arange(len(intras_normalized[idx])), intras_normalized[idx], color='black')
        axes[i].plot(np.arange(len(preds[idx])), preds[idx], color='red')
    
    plt.tight_layout()  # Adjust layout to make sure everything fits without overlap
    
    # If a save directory is provided, save the figure
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # Create the directory if it doesn't exist
        save_path = os.path.join(save_dir, f'{name}.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')  # Save as SVG with tight bounding box
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_physics( intras_test_,v_out, dv_out_max,dv_t,dv2_t,a_oot,k_oot,x_oot,t_oot,dv_l,dv_r,t_out_broad, save_path,  name ='results'):
            v_out = v_out.reshape(-1,8000)
            dv_out_max= dv_out_max.reshape(dv_out_max.shape[0],-1)
            dv_t = dv_t.reshape(dv_t.shape[0],-1)
            dv2_t= dv2_t.reshape(dv2_t.shape[0],-1)
            a_oot = a_oot.reshape(a_oot.shape[0],-1)
            k_oot=  k_oot.reshape(k_oot.shape[0],-1)
            x_oot=  x_oot.reshape(x_oot.shape[0],-1)
            t_oot = t_oot.reshape(t_oot.shape[0],-1)
            # print(a_oot,x_oot,k_oot)
            pred_data = []
            intra_data = []
            ap_data = []
            dvts = []
            dvvtts = []
            for i in range(0,len(dv_out_max),5):
                v = v_out[i]
                dvt = dv_t[i]
                dvtt=dv2_t[i]
                a = a_oot[i][0]
                k = k_oot[i][0]
                x = x_oot[i][0]
                t = t_oot[i][0]
                print('a,k,x,t',a,k,x,t)

                t_eval = np.linspace(0, t, 8000)
                t_span = (0, t)  
                solution_opt = solve_ivp(aliev_panfilov_ode, t_span, [0.1, 0.3], args=(a, k, x), t_eval=t_eval, method='RK45')
                ap_rec = solution_opt.y[0]
        
                plt.plot(v, label = 'pred', color = 'red')
                plt.plot(intras_test_[i], label ='intra', color = 'black')
                plt.plot(ap_rec/np.max(ap_rec), label ='ap', color = 'green')
                pred_data.append(v)
                intra_data.append(v_out[i])
                ap_data.append(ap_rec)
                dvts.append(dvt)
                dvvtts.append(dvtt)
            plt.figure()
            for v, intra in zip(pred_data, intras_test_):
                plt.plot(v, label='pred', color='red', alpha=0.3)
                plt.plot(intra, label='intra', color='black', alpha=0.3)
            for  ap_rec in ap_data:
                plt.plot(ap_rec/np.max(ap_rec), label='ap', color='green', alpha=0.3)
            plt.savefig(f"{save_path}/{name}.png", format='png', dpi=300)
            plt.show()
            
            plt.figure()
            for app in dvts:
                plt.plot(app, color='green')
            plt.show()

            for app in dvvtts:
                plt.plot(app, color='green')
            plt.show()
