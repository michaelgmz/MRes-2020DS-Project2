# %%
# ----------------------------------------------------------------
# Parameter List will be in the following order (19 items):
# [Acceleration, Acceleration X, Acceleration Y, Acceleration Z, 
#  Displacement X, Displacement Y, Displacement Z, Displacement^2, 
#  Distance to Image Border XY, Distance to Image Border XYZ, Position X, 
#  Position Y, Position Z, Speed, Time, Time Index, Velocity X, 
#  Velocity Y, Velocity Z]
# PLEASE just ignore the undefined variables warning.
# Cell Index starts with 0 Time Frame starts with 1 Items starts with 0
# ----------------------------------------------------------------
import dill
import os
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
os.chdir('D:\\Rotation2\\IVM-MRes project #2\\IVM machine learning project 2020 (Mingze)\\Data xlsx')
dill.load_session('GlobalVariable.pkl')
del In, Out, all_files

# %%
# Visualise cell position of a single experiment of 60 seconds
# Function not finished need further pre-processing
def Draw_Cell_Position(Observation):
    # Identify Cell Index & Time Frame
    num_of_cells = len(Observation)
    if Observation[0][61] == []:
        max_frame = 60
    else: 
        max_frame = 61
    
    x_axis = []
    y_axis = []
    for index in range(num_of_cells):
        x = []
        y = []
        for frame in range(1, max_frame + 1):
            if Observation[index][frame] != []:
                x.append(Observation[index][frame][10])
                y.append(Observation[index][frame][11])
                x_axis.append(Observation[index][frame][10])
                y_axis.append(Observation[index][frame][11])
        plt.plot(x, y, linewidth = 0.5)
    x_min, x_max, y_min, y_max = min(x_axis), max(x_axis), min(y_axis), max(y_axis)
    plt.axis([25, 250, 25, 275])

    plt.savefig('fty720_0_60.svg', format = 'svg')

# %%
Draw_Cell_Position(fty720_0_60)