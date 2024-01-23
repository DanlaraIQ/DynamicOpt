import numpy as np
import matplotlib.pyplot as plt
from mealpy import FloatVar, DE, GWO, CSA
import time

# Define system dynamics and objective function calculation


def system_dynamics(x, params):
    # Calculation of dimensionless temperatures
    y_c = params['T_c'] / (params['J'] * params['c_f'])
    y_f = params['T_f'] / (params['J'] * params['c_f'])

    # Time discretization for simulation
    time = np.linspace(params['t0'], params['tf'], params['N_points'])
    dt = (params['tf'] - params['t0']) / (params['N_points'] - 1)

    # Initialize arrays for y1, y2, and control variable u
    y1 = np.zeros(params['N_points'])
    y2 = np.zeros(params['N_points'])
    u = x
    u[0] = params['u_ini']
    y1[0] = params['y1_ini']
    y2[0] = params['y2_ini']

    # Calculate dynamics and objective function over time
    total = 0
    for i in range(params['N_points'] - 1):
        # Dynamical equations
        y1[i + 1] = y1[i] + dt * ((1 - y1[i]) / params['theta'] - params['k_10'] * np.exp(-params['N'] / y2[i]) * y1[i])
        y2[i + 1] = y2[i] + dt * ((y_f - y2[i]) / params['theta'] + params['k_10'] *
                                  np.exp(-params['N'] / y2[i]) * y1[i] - params['alpha'] * u[i] * (y2[i] - y_c))

        # Increment objective function value
        total += params['alpha1'] * (params['y1_final'] - y1[i])**2 + params['alpha2'] * \
            (params['y2_final'] - y2[i])**2 + params['alpha3'] * (params['u_final'] - u[i])**2

    # Final objective function value
    obj_func = total * dt / 3
    return y1, y2, obj_func


def objective_function(x, params):
    # Return only the final objective function value
    _, _, obj_func = system_dynamics(x, params)
    return obj_func


# Define different states for initial and final conditions
states = {
    'A': {'y1_ini': 0.0944, 'y2_ini': 0.7766, 'u_ini': 340},
    'B': {'y1_ini': 0.1367, 'y2_ini': 0.7293, 'u_ini': 390},
    'C': {'y1_ini': 0.1926, 'y2_ini': 0.6881, 'u_ini': 430},
    'D': {'y1_ini': 0.2632, 'y2_ini': 0.6519, 'u_ini': 455},
}

# Function to run optimizer and measure execution time


def run_optimizer(optimizer_class, **kwargs):
    start_time = time.time()
    optimizer = optimizer_class(**kwargs)
    optimizer.solve(problem_dict)
    end_time = time.time()
    execution_time = end_time - start_time
    return optimizer.g_best.solution, execution_time

# Function to optimize transition between states and update parameters


def optimize_transition(initial_state, final_state):
    parameters.update(states[initial_state])
    parameters['y1_final'] = states[final_state]['y1_ini']
    parameters['y2_final'] = states[final_state]['y2_ini']
    parameters['u_final'] = states[final_state]['u_ini']

    problem_dict['bounds'] = FloatVar(lb=[0] * parameters['N_points'], ub=[1000] * parameters['N_points'])

    best_solution_CSA, time_CSA = run_optimizer(CSA.OriginalCSA, epoch=5, pop_size=50, p_a=0.3)
    best_solution_DE, time_DE = run_optimizer(DE.OriginalDE, epoch=500, pop_size=50, wf=0.7, cr=0.9, strategy=1)
    best_solution_GWO, time_GWO = run_optimizer(GWO.OriginalGWO, epoch=5, pop_size=50)

    return best_solution_CSA, best_solution_DE, best_solution_GWO, time_CSA, time_DE, time_GWO

# Function to plot and save the results


def plot_and_save_results(y_values_DE, y_values_GWO, y_values_CSA, time_points, labels, title, ylabel, plot_name):
    plt.figure(figsize=(5, 4))
    plt.plot(time_points, y_values_DE, label=labels[0])
    plt.plot(time_points, y_values_GWO, label=labels[1])
    plt.plot(time_points, y_values_CSA, label=labels[2])
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_name}.png")
    plt.show()


# Main execution
if __name__ == "__main__":
    # Choose initial and final states (ensure they are different)
    initial_state = 'B'
    final_state = 'A'

    # Combine initial state, final state, and other parameters
    parameters = {
        'c_f': 7.6, 'T_f': 300, 'T_c': 290, 'J': 100, 'alpha': 1.95e-04,
        'k_10': 300, 'N': 5, 'theta': 20,
        't0': 0, 'tf': 10, 'alpha1': 1e+06, 'alpha2': 2e+03, 'alpha3': 1e-03,
        'N_points': 20
    }

    problem_dict = {
        "obj_func": lambda x: objective_function(x, parameters),
        "minmax": "min"
    }

    # Run optimization for the chosen transition
    best_solution_CSA, best_solution_DE, best_solution_GWO, time_CSA, time_DE, time_GWO = optimize_transition(
        initial_state, final_state)

    # Time points for plotting
    time_points = np.linspace(parameters['t0'], parameters['tf'], parameters['N_points'])

    # Obtain dynamics for plotting
    y1_CSA, y2_CSA, _ = system_dynamics(best_solution_CSA, parameters)
    y1_DE, y2_DE, _ = system_dynamics(best_solution_DE, parameters)
    y1_GWO, y2_GWO, _ = system_dynamics(best_solution_GWO, parameters)

    # Plot and save results for y1, y2, and u
    plot_and_save_results(y1_DE, y1_GWO, y1_CSA, time_points, [
                          "DE", "GWO", "CSA"], "Comparison of y1 Evolution", "$y_1$", "Comparison_y1")
    plot_and_save_results(y2_DE, y2_GWO, y2_CSA, time_points, [
                          "DE", "GWO", "CSA"], "Comparison of y2 Evolution", "$y_2$", "Comparison_y2")
    plot_and_save_results(best_solution_DE, best_solution_GWO, best_solution_CSA, time_points, [
                          "DE", "GWO", "CSA"], "Comparison of Control Variable (u) Evolution", "Control Variable (u)", "Comparison_u")
    print(y1_DE, y2_DE, best_solution_DE)
