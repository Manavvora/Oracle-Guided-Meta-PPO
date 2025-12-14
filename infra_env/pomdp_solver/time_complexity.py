import os
import time
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle

def run_script(command):
    start_time = time.perf_counter()
    cpu_start_time = time.process_time()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {command[1]}: {e}")
        return None  # Indicate failure by returning None
    end_time = time.perf_counter()
    cpu_end_time = time.process_time()
    return end_time - start_time, cpu_end_time - cpu_start_time

def run_all_scripts(num_components):
    scripts = [
        'random_forest.py',
        'random_forest_budget_split.py',
        'generate_oracle_policies_optimal_budget_split.py',
        'oracle_guided_meta_ppo_optimal_budget_split.py'
    ]

    commands = []
    for script in scripts:
        if script == 'random_forest.py':
            commands.append(['python3', script])
        else:
            commands.append(['python3', script, '--num_components', str(num_components)])

    times_taken = []
    cpu_times_taken = []
    for command in commands:
        script_name = command[1]
        time_taken, cpu_time_taken = run_script(command)
        if time_taken is not None:
            times_taken.append(time_taken)
            cpu_times_taken.append(cpu_time_taken)
            print(f'{script_name} took {time_taken:.10f} seconds')
            print(f'{script_name} took {cpu_time_taken:.10f} seconds of CPU time')
        else:
            times_taken.append(0)
            cpu_times_taken.append(0)
            print(f'{script_name} failed to run')

    return times_taken, cpu_times_taken

num_components_list = [1, 2, 5, 10, 20, 50, 100, 500, 1000]
print(f'Running all scripts for num_components = {num_components_list}')
script_names = ['random_forest', 'budget_split', 'value_iteration_mdp', 'meta_ppo_pomdp']
times_taken_per_script = {script: [] for script in script_names}
cpu_times_taken_per_script = {script: [] for script in script_names}
total_times_taken = []
total_cpu_times_taken = []

for num_components in num_components_list:
    print(f'Running all scripts for num_components = {num_components}')
    times_taken, cpu_times_taken = run_all_scripts(num_components)
    total_time_taken = sum(times_taken)
    total_cpu_time_taken = sum(cpu_times_taken)
    total_times_taken.append(total_time_taken)
    total_cpu_times_taken.append(total_cpu_time_taken)
    for script, time_taken, cpu_time_taken in zip(script_names, times_taken, cpu_times_taken):
        times_taken_per_script[script].append(time_taken)
        cpu_times_taken_per_script[script].append(cpu_time_taken)
    print(f'Total time for {num_components} components: {total_time_taken:.2f} seconds')
    print(f'Total CPU time for {num_components} components: {total_cpu_time_taken:.2f} seconds')

with open('results/time_complexity_analysis/times_taken_per_script.pkl', 'wb') as f:
    pickle.dump(times_taken_per_script, f)

with open('results/time_complexity_analysis/cpu_times_taken_per_script.pkl', 'wb') as f:
    pickle.dump(cpu_times_taken_per_script, f)

np.save('results/time_complexity_analysis/total_times_taken.npy', total_times_taken)
np.save('results/time_complexity_analysis/total_cpu_times_taken.npy', total_cpu_times_taken)

# Plot the results for individual scripts
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

colors = sns.color_palette("husl", len(script_names))

for idx, (script, color) in enumerate(zip(script_names, colors)):
    sns.lineplot(x=np.log10(num_components_list), y=times_taken_per_script[script], ax=axes[idx], color=color, marker='o', label=f'Time Taken for {script}')
    axes[idx].set_title(f'Time Taken for {script}')
    axes[idx].set_ylabel('Time (seconds)')
    axes[idx].grid(True)
    axes[idx].legend()

for ax in axes:
    ax.set_xlabel('Log of Number of Components')

fig.suptitle('Computational Complexity Analysis per Script (Wall-Clock Time)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/plots/computational_complexity_per_script.png')
plt.show()

# Plot the total times
plt.figure(figsize=(10, 6))
sns.lineplot(x=np.log10(num_components_list), y=total_times_taken, label='Total Time Taken', color='m', marker='o')


plt.xlabel('Log of Number of Components')
plt.ylabel('Total Time Taken (seconds)')
plt.title('Total Computational Complexity Analysis (Wall-Clock Time)')
plt.legend()
plt.grid(True)
plt.savefig('results/plots/computational_complexity_total.png')
plt.show()

# Plot the results for individual scripts CPU times
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

for idx, (script, color) in enumerate(zip(script_names, colors)):
    sns.lineplot(x=np.log10(num_components_list), y=cpu_times_taken_per_script[script], ax=axes[idx], color=color, marker='o', label=f'CPU Time Taken for {script}')
    axes[idx].set_title(f'CPU Time Taken for {script}')
    axes[idx].set_ylabel('Time (seconds)')
    axes[idx].grid(True)
    axes[idx].legend()

for ax in axes:
    ax.set_xlabel('Log of Number of Components')

fig.suptitle('Computational Complexity Analysis per Script (CPU Time)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('results/plots/computational_complexity_per_script_cpu.png')
plt.show()

# Plot the total CPU times
plt.figure(figsize=(10, 6))
sns.lineplot(x=np.log10(num_components_list), y=total_cpu_times_taken, label='Total CPU Time Taken', color='m', marker='o')

plt.xlabel('Log of Number of Components')
plt.ylabel('Total CPU Time Taken (seconds)')
plt.title('Total Computational Complexity Analysis (CPU Time)')
plt.legend()
plt.grid(True)
plt.savefig('results/plots/computational_complexity_total_cpu.png')
plt.show()