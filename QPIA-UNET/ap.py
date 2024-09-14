from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.model_selection import ParameterGrid
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Define the epsilon function
def epsilon(u, a, x):
    return x if u >= a else (1-x)

# Define the system of ODEs
def aliev_panfilov(Y, t, a, k, x):
    u, v = Y
    du_dt = k * u * (1 - u) * (u - a) - u * v
    dv_dt = epsilon(u, a, x) * (k * u - v)
    return [du_dt, dv_dt]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def epsilon_smooth(u, a, x):
    step = x * sigmoid(10e5 * (u - a)) + (sigmoid(10e5 * (a - u))) * (1 - x)
    return step

def aliev_panfilov_ode(t, y, a, k, x):
    u = y[0]
    du_dt = y[1]
    # Calculate second derivative using your equation
    d2u_dt2 = -epsilon_smooth(u, a, x) * (k * u * (u**2 + a - a * u) + du_dt) + (k * u * (1 - 2 * u + a) * du_dt + (1/u) * du_dt**2)
    
    return [du_dt, d2u_dt2]


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
                solution_opt = solve_ivp(aliev_panfilov_ode, t_span, [0.1, 0.05], args=(a, k, x), t_eval=t_eval, method='RK45')
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
            for v, intra, ap_rec in zip(pred_data, intras_test_, ap_data):
                plt.plot(v, label='pred', color='red', alpha=0.3)
                plt.plot(intra, label='intra', color='black', alpha=0.3)
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


