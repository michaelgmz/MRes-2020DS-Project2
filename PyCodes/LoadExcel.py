# %%
'''----------------------------------------------------------------
There are two types of data formar .csv and .xlsx files
This script deals with .xlsx files only
All imports goes here
----------------------------------------------------------------'''
import os
import pandas as pd
import re
import dill
import pickle

# %% 
'''----------------------------------------------------------------
Get and iterate over all file paths
----------------------------------------------------------------'''
def Get_Data_Path(objective_directory):
    all_files = []
    for lists in os.listdir(objective_directory):
        if lists.endswith('.xlsx'):
            all_files.append(os.path.join(objective_directory, lists))

    return all_files

# %%
'''----------------------------------------------------------------
General explore and import of single .xlsx file
Sheetnames fixed
----------------------------------------------------------------'''
def Import_Single_Excel(file_path):
    PossibleSheets = ['Overall', 'Acceleration', 'Acceleration Vector', 'Displacement', 
                        'Displacement^2', 'Distance to Image Border XY', 'Distance to Image Border XYZ', 
                        'Position', 'Speed', 'Time', 'Time Index', 'Track Duration', 'Velocity']
    wb = pd.ExcelFile(file_path)
    sheets = wb.sheet_names

    # Get total number of cells
    ws = pd.read_excel(wb, sheet_name = 'Track Duration', header = 1)
    cell_number, tracking_time = Cell_Tracking_Time(worksheet = ws)

    # Generate cell info hash dictionary
    # Excel file -> Cell Index -> Time frame -> Detailed Info
    cell_details = {index:{} for index in range(cell_number)}
    for key in cell_details.keys():
        cell_details[key] = {frame:[] for frame in range(1, 62)}
        cell_details[key]['Track Duration'] = tracking_time[key]

    # Process the rest of sheets to get cell info
    for sheet in sheets:
        # print (f'----- Executing Sheet [{sheet}] -----\n')
        assert sheet in PossibleSheets, 'Sheet Name NOT Valid.'
        ws = pd.read_excel(wb, sheet_name = sheet, header = 1)
        # print (f'----- Sheet Header: {list(ws)} -----\n')

        if sheet == 'Overall':
            total = ws['Value'][0]
            # print (f'----- Total Number of Spots is {total} -----\n')
        
        elif sheet == 'Track Duration':
            continue

        else:
            cell_details = Generate_Cell_Info(worksheet = ws, cell_details = cell_details, sheet = sheet)

    return cell_details

# %%
'''----------------------------------------------------------------
Generate Cell Info based on all the sheets (except Overall & Track Duration)
The result would be a multi-layered dictionary for each individual cell
----------------------------------------------------------------'''
def Generate_Cell_Info(worksheet, cell_details, sheet):
    '''
    Each dictionary will have same number of (keys & values) as cells
    Each cell will have 61 time frames
    Some cell may have NULL NaN values obviously
    Each key will be a time frame whilst each value will be a list of parameters according to that time frame
    Parameter List will be in the following order (19 items):
    [Acceleration, Acceleration X, Acceleration Y, Acceleration Z, Displacement X, Displacement Y, Displacement Z, 
    Displacement^2, Distance to Image Border XY, Distance to Image Border XYZ, Position X, Position Y, Position Z, 
    Speed, Time, Time Index, Velocity X, Velocity Y, Velocity Z]'''

    # Sheet with 1 value to be added goes here
    if sheet in ['Acceleration', 'Displacement^2', 'Distance to Image Border XY', 'Distance to Image Border XYZ', 'Speed', 'Time', 'Time Index']:
        for index, row in worksheet.iterrows():
            cell_index = row['TrackID'] - 1000000000
            time_frame = row['Time']
            cell_details[cell_index][time_frame].append(row['Value'])

    # Sheet with 3 values to be added goes here
    elif sheet in ['Acceleration Vector', 'Displacement', 'Position', 'Velocity']:
        for index, row in worksheet.iterrows():
            cell_index = row['TrackID'] - 1000000000
            time_frame = row['Time']
            for i in range(3):
                cell_details[cell_index][time_frame].append(row[i])

    else:
        raise NameError('Sheet name not recognized.')
    return cell_details

# %%
'''----------------------------------------------------------------
Process Sheet (Track Duration)
Total cell number & time tracked in a dictionary format
----------------------------------------------------------------'''
def Cell_Tracking_Time(worksheet):
    tracking_time = {}
    cell_number = len(worksheet['ID'])
    # print (f'----- Total Number of Cells is {cell_number} -----\n')

    worksheet_sorted = worksheet.sort_values(by = 'ID')
    for index, row in worksheet_sorted.iterrows():
        if row['ID'] not in tracking_time:
            tracking_time[row['ID'] - 1000000000] = row['Value']

    return cell_number, tracking_time

# %%
'''----------------------------------------------------------------
Intermediate Function | pass individual .xslx file for processing
----------------------------------------------------------------'''
def Processing_Single_Excel(file_path):
    '''
    Iterately single Excel file (experiment & time intervals) processing
        Check if CD11a was injected in the experiment (file name)
        If it exists, LEAVE the file FOR NOW
        Use none CD11a experiments
    Without B cells in file name ---> ONLY neutrophils
    With B cells in file name ---> ONLY B cells
    Three groups: 
    CD11a & CD49d | FTY720 | PBS & AMD3100 (Ly6G_CD31_CD169)'''
    
    file_name = os.path.split(file_path)[1]
    print (f'---> Processing {file_name}\n')
    single_file_data = Import_Single_Excel(file_path = file_path)

    # if 'cd11a' in file_path:
    #     print (f'----- Processing {file_name} -----\n')
    #     variable = 'cd11cd49_' + re.findall(".*-19 (.*) frames.*", file_name)[0]
    #     single_file_data = Import_Single_Excel(file_path = file_path)

    # elif 'FTY720' in file_path:
    #     print (f'----- Processing {file_name} -----\n')
    #     variable = 'fty720_' + re.findall(".*-19 (.*) frames.*", file_name)[0]
    #     single_file_data = Import_Single_Excel(file_path = file_path)

    # else:
    #     print (f'----- Processing {file_name} -----\n')
    #     if 'AMD3100' in file_path:
    #         variable = 'amd3100_' + re.findall(".*spleen (.*) frames.*", file_name)[0]
    #     else:
    #         variable = 'pbs_' + re.findall(".*spleen (.*) frames.*", file_name)[0]
    #     single_file_data = Import_Single_Excel(file_path = file_path)

    return single_file_data, file_name

# %%
'''----------------------------------------------------------------
Main part of the module goes here
For each file and subsequently each cell 
Besides 1-60 or 61 that will be the details for each time frame
Time Frame Index ['Track Duration'] will be the tracking_time
----------------------------------------------------------------'''
print (f'----- Loading Data -----\n')
EXPERIMENT_PATH = 'D:\Rotation2\VideoFrame\Exp 18-5-18 FTY720'
all_files = Get_Data_Path(EXPERIMENT_PATH)

for file_path in all_files:
    single_file_data, file_name = Processing_Single_Excel(file_path)
    pickle.dump(single_file_data, open(os.path.join(os.path.split(file_path)[0], file_name[:-4] + 'pkl'), 'wb'))
print (f'----- Data Loaded Successfully -----\n')