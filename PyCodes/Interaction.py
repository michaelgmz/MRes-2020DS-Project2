# %%
'''----------------------------------------------------------------
This script deals with Neutrophils and B cells interaction
----------------------------------------------------------------'''
import os
import pickle
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')
import objecttrack
N_PATH = 'D:\Rotation2\VideoFrame\SecondCD11a'
B_PATH = 'D:\Rotation2\VideoFrame\Exp 19-4-18 CD11a blocking'
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = 1002, 1002, 3

# %%
'''----------------------------------------------------------------
self.id, set, may contain multiple ids
self.hits, int, # of frames the cell has been hit
self.pat, dict, {frame # : [intensity, volume, smooth, circumference, centre, 
    (startX, startY, endX, endY), velocity, acceleration, displacement]}
self.cont, np.array, {frame # : contour}, contour of the detected cell
----------------------------------------------------------------'''
N_before = pickle.load(open(os.path.join(N_PATH, 'ncellbefore.pkl'), 'rb'))
N_after = pickle.load(open(os.path.join(N_PATH, 'ncellafter.pkl'), 'rb'))

# %%
'''----------------------------------------------------------------
Read B cells information | ExcelFile data
----------------------------------------------------------------'''
def ReadBFile(B_PATH, label):
    for file in os.listdir(B_PATH):
        if file.endswith('.xlsx') and 'B' in file:
            if label in file:
                B = pd.ExcelFile(os.path.join(B_PATH, file))

    return B

# %%
'''----------------------------------------------------------------
Distance from every primary cell to the closest secondary cell at every frame
Plot against the speed of primary cell
----------------------------------------------------------------'''
def NBInteraction(N, label):
    B = ReadBFile(B_PATH, label)
    b_position = pd.read_excel(B, sheet_name = 'Position', header = 1)
    b_speed = pd.read_excel(B, sheet_name = 'Speed', header = 1)

    if label == 'before':
        b_position['Position X'] = b_position['Position X'] * 1.7105 - 983.6
        b_position['Position Y'] = b_position['Position Y'] * (-1.7098) + 1955.9
    if label == 'after':
        b_position['Position X'] = b_position['Position X'] * 1.7395 - 1007.4
        b_position['Position Y'] = b_position['Position Y'] * (-1.7216) + 1964.7

    df_proximity = pd.DataFrame(columns = [str(column) for column in range(1, 61)])
    df_speed = pd.DataFrame(columns = [str(column) for column in range(1, 61)])
    for cell in tqdm(N):
        if cell.hits >= 3:
            if len(cell.id) == 1:
                n_pattern = cell.pat
                n_contour = cell.cont
                df_proximity.loc[list(cell.id)[0]] = [-1] * 60
                df_speed.loc[list(cell.id)[0]] = [-1] * 60

                for frame in n_pattern.keys():
                    frame_b_position = b_position[b_position['Time'] == frame]
                    frame_b_speed = b_speed[b_speed['Time'] == frame]
                    n_speed, n_cont = n_pattern[frame][6], n_contour[frame]

                    b_id = []
                    n_b_speed = [round(n_speed, 4)]
                    for index, row in frame_b_position.iterrows():
                        b_centre = (int(row['Position X']), int(row['Position Y']))
                        img_b = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype = np.uint8)
                        img_b = cv2.circle(img_b, b_centre, int(4 / 0.481), 255, -1)

                        img_n = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype = np.uint8)
                        img_n = cv2.drawContours(img_n, [n_cont], -1, 1)
                        intersection = np.logical_and(img_n, img_b)
                        
                        if intersection.any() == True:
                            b_id.append(row['TrackID'] - 1000000000)
                            n_b_speed.append(round(frame_b_speed.loc[index, 'Value'], 4))

                    df_proximity.at[list(cell.id)[0], str(frame)] = b_id
                    df_speed.at[list(cell.id)[0], str(frame)] = n_b_speed
    
    return df_proximity, df_speed

# %%
'''----------------------------------------------------------------
Main function | Gather proximity and speed data
----------------------------------------------------------------'''
proximity_before, speed_before = NBInteraction(N_before, 'before')
proximity_after, speed_after = NBInteraction(N_after, 'after')

pickle.dump(proximity_before, open(os.path.join(N_PATH, 'proximitybefore.pkl'), 'wb'))
pickle.dump(proximity_after, open(os.path.join(N_PATH, 'proximityafter.pkl'), 'wb'))
pickle.dump(speed_before, open(os.path.join(N_PATH, 'speedbefore.pkl'), 'wb'))
pickle.dump(speed_after, open(os.path.join(N_PATH, 'speedafter.pkl'), 'wb'))

# %%
'''----------------------------------------------------------------
Main function | Load proximity and speed data
----------------------------------------------------------------'''
p_before = pickle.load(open(os.path.join(N_PATH, 'proximitybefore.pkl'), 'rb'))
p_after = pickle.load(open(os.path.join(N_PATH, 'proximityafter.pkl'), 'rb'))
s_before = pickle.load(open(os.path.join(N_PATH, 'speedbefore.pkl'), 'rb'))
s_after = pickle.load(open(os.path.join(N_PATH, 'speedafter.pkl'), 'rb'))
labels = pickle.load(open(os.path.join(N_PATH, '197181.pkl'), 'rb'))

# %%
'''----------------------------------------------------------------
N-B Pairwise speed dataframe
----------------------------------------------------------------'''
import pandas as pd
def ns_vs_bs(df_speed, label):
    df_speed_pair = pd.DataFrame(columns = ['N', 'B', 'Cluster'])
    count = 1

    row_index = 0
    for index, row in df_speed.iterrows():
        for i in range(1, 61):
            if row[str(i)] != -1:
                if len(row[str(i)]) > 1:
                    ns = row[str(i)][0]
                    bs = row[str(i)][1:]

                    for item in bs:
                        df_speed_pair.at[count, 'N'] = ns
                        df_speed_pair.at[count, 'B'] = item
                        df_speed_pair.at[count, 'Cluster'] = label[row_index]
                        count += 1
        row_index += 1
    
    return df_speed_pair

speed_pair_before = ns_vs_bs(s_before, labels[:197])
speed_pair_after = ns_vs_bs(s_after, labels[197:])

# %%
'''----------------------------------------------------------------
Gamma Distribution?
----------------------------------------------------------------'''
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma
plt.style.use('ggplot')

def gamma_plot(cell, savefig = False):
    data = pd.concat([speed_pair_before, speed_pair_after])
    data = list(data[data[cell] > 0][cell])

    x = np.linspace(0, 35, 1000)
    shape, loc, scale = gamma.fit(data, floc = 0)
    print (f'{cell} Expected: {shape * scale}')

    if savefig:
        sns.distplot(data, kde = False, norm_hist = True, color = '#3498db', bins = np.linspace(0, 8, 32))
        y = gamma.pdf(x, shape, loc, scale)

        plt.title(f'Distribution of {cell} cells\' speed', fontsize = 15)
        plt.axvline(shape * scale, linestyle = 'dashed', color = 'black', zorder = 1, 
            label = 'Expectation', linewidth = 3)
        plt.xlim([0, 8])
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.xlabel(f'Velocity ({chr(956)}m/s)', fontsize = 12)
        plt.plot(x, y, label = 'Fitted Gamma', linewidth = 3)

        leg = plt.legend(prop = {'size': 12})
        for line in leg.get_lines():
            line.set_linewidth(3)
        plt.tight_layout()
        plt.savefig(os.path.join(N_PATH, f'{cell}-Gamma.png'), format = 'png', dpi = 300)
        plt.savefig(os.path.join(N_PATH, f'{cell}-Gamma.pdf'), format = 'pdf', dpi = 500)
        plt.close()

    return shape * scale

n_expection = gamma_plot('N', savefig = False)
b_expection = gamma_plot('B', savefig = False)

# %%
'''----------------------------------------------------------------
Scatter Plot?
----------------------------------------------------------------'''
def interact_scatter(data, treatment):
    inner = data[data['N'] < n_expection]
    inner = inner[inner['B'] < b_expection]
    outer = pd.concat([data, inner]).drop_duplicates(keep = False)

    fig = plt.figure()
    clist = ['salmon', '#3498db']
    for index, frame in enumerate([inner, outer]):
        if index == 0:
            label = f'# of interactions {frame.shape[0]}'
        else:
            label = f'# of non-interactions {frame.shape[0]}'
        plt.scatter(frame['N'], frame['B'], color = clist[index], s = 3, label = label)

    plt.xlim([-0.05, 6])
    plt.ylim([-0.05, 6])
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel(f'Neutrophil velocity ({chr(956)}m/s)', fontsize = 15)
    plt.ylabel(f'B cell velocity ({chr(956)}m/s)', fontsize = 15)
    plt.title(f'{treatment} Treatment', fontsize = 15)
    leg = plt.legend(prop = {'size': 15}, markerscale = 3)
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(N_PATH, 'Scatter.png'), format = 'png', dpi = 300)
    # plt.savefig(os.path.join(N_PATH, 'Scatter.pdf'), format = 'pdf', dpi = 300)
    # plt.close()

interact_scatter(speed_pair_before, 'Before')
interact_scatter(speed_pair_after, 'After')

# %%
'''----------------------------------------------------------------
Hist Plot?
----------------------------------------------------------------'''
def interact_hist(df_proximity, df_speed, color, label, show_img = False):
    interact_count = []
    df_temp = pd.DataFrame(columns = [str(column) for column in range(1, 61)])

    for index, p_row in df_proximity.iterrows():
        s_row = df_speed.loc[index]
        temp_count = 0
        df_temp.loc[index] = p_row

        for i in range(1, 61):
            b_indexs = []
            control = False

            if p_row[str(i)] != -1 and p_row[str(i)] != []:
                if s_row[str(i)][0] <= n_expection:
                    for n, b in enumerate(s_row[str(i)][1:]):
                        if b <= b_expection:
                            temp_count += 1
                            b_indexs.append(p_row[str(i)][n])
                        else:
                            control = True
                else:
                    control = True

            if control:
                df_temp.at[index, str(i)] = []
            if len(b_indexs) > 0:
                df_temp.at[index, str(i)] = b_indexs
        
        interact_count.append(temp_count)
    
    # df_temp['Cluster'] = labels[:197] if 'Before' in label else labels[197:]

    if show_img:
        sns.distplot(interact_count, kde = True, color = color, bins = [i for i in range(0, 61, 3)], 
            label = label)
        plt.xlim([6, 60])
        plt.ylim([0, 0.03])
        plt.xlabel('# of interactions per neutrophil across 60 frames', fontsize = 15)
        plt.ylabel('Proportion / Bin size', fontsize = 15)
        plt.title('Neutrophil interaction summary', fontsize = 16)
        plt.legend()
        plt.tight_layout()

    return interact_count, df_temp

count_before, inter_before = interact_hist(p_before, s_before, 'salmon', 'Before Treatment', show_img = False)
count_after, inter_after = interact_hist(p_after, s_after, '#3498db', 'After Treatment', show_img = False)
pickle.dump(inter_before, open(os.path.join(B_PATH, 'interbefore.pkl'), 'wb'))
pickle.dump(inter_after, open(os.path.join(B_PATH, 'interafter.pkl'), 'wb'))

# %%
'''----------------------------------------------------------------
Length of interactions?
----------------------------------------------------------------'''
from itertools import groupby
def interact_length(df_inter, n_class):
    interaction_seq, appear_seq, interaction_len = [], [], []

    row_control = 0
    for index, row in df_inter.iterrows():
        interaction_count = 0
        appear_count = 0
        total_count = 0

        for i in range(1, 61):
            if row[str(i)] != -1:
                appear_count += 1
                if row[str(i)] != []:
                    interaction_count += 1
                    total_count += len(row[str(i)])
        
        inter_per_class[n_class[row_control]].append(total_count)
        row_control += 1
        
        items, sums = [], []
        for item, group in groupby(list(df_inter.loc[index])):
            items.append(item if item != -1 and len(item) > 0 else -1)
            sums.append(sum(1 for sub in group) if items[-1] != -1 else 0)
        
        temp_i, temp_s, res_i, res_s = [], [], [], []
        items.append(-1)
        for n, i in enumerate(items):
            if i != -1:
                temp_i.append(i)
                temp_s.append(sums[n])
            else:
                if temp_i == []:
                    pass
                else:
                    res_i.append(temp_i)
                    res_s.append(temp_s)
                    temp_i, temp_s = [], []
        
        for times in res_s:
            if len(times) == 1:
                interaction_len.append(times[0])
            else:
                interaction_len.append(sum(times))
                interaction_len.append(max(times) if len(times) == 2 else sum(times[1:]))

        interaction_seq.append(interaction_count)
        appear_seq.append(appear_count)

    return [interaction_seq[i] / appear_seq[i] for i in range (len(interaction_seq))], \
        interaction_len

inter_per_class = {0:[], 1:[]}
propor_b, len_b = interact_length(inter_before, labels[:197])
propor_a, len_a = interact_length(inter_after, labels[197:])

# %%
# Normality | Non-parametric tests
import scipy.stats as stats
statistics_b, pvals = stats.shapiro(propor_b)
print (f'p-value (Shapiro Before): {pvals}, S: {statistics_b}')
print (f'df: {len(propor_b)}')

statistics_a, pvals = stats.shapiro(propor_a)
print (f'p-value (Shapiro After): {pvals}, S: {statistics_a}')
print (f'df: {len(propor_a)}')

statistics_u, pvals = stats.mannwhitneyu(propor_b, propor_a, alternative = 'less')
print ('p-value (Mann-Whitney U test): \t', pvals)
print (f'Before: {np.median(propor_b)}, After: {np.median(propor_a)}')
print (f'U: {statistics_u}')

# %%
df = pd.DataFrame(columns = ['Proportion', 'Treatment', 'Color'])
df['Treatment'] = ['Dataset 1'] * len(propor_b) + ['Dataset 2'] * len(propor_a)
df['Color'] = ['#3498db'] * len(propor_b) + ['#2ecc71'] * len(propor_a)
propor_b.extend(propor_a)
df['Proportion'] = propor_b

# %%
# Interaction # 1
import seaborn as sns
fig = sns.boxplot(x = 'Treatment', y = 'Proportion', data = df, palette = ['#3498db', '#2ecc71'])
fig.set_xlabel('Treatment', fontsize = 15)
fig.set_ylabel('Proportion', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.title('Interaction time / Neutrophil existing time', fontsize = 15)
plt.tight_layout()
plt.savefig(os.path.join(N_PATH, 'InteractionNo1.png'), format = 'png', dpi = 300)
plt.savefig(os.path.join(N_PATH, 'InteractionNo1.pdf'), format = 'pdf', dpi = 500)
plt.close()

# %%
# Interaction # 3
import scipy.stats as stats
obs = np.empty((2, 4))
obs = [[1831, 288, 16, 2],
        [int(1280 * 2137 / 1414), int(124 * 2137 / 1414), 
        int(9 * 2137 / 1414), int(1 * 2137 / 1414)]]
chi2, p, dof, expected = stats.chi2_contingency(obs)
print (f'Sample size: {np.sum(obs)}')
print (f'Chi-square： {chi2}, p value: {p}, dof: {dof}, expected: {expected}')

# %%
# Interaction # 2
from collections import Counter
N = max(max(len_b), max(len_a))
count_b = Counter(len_b)
count_a = Counter(len_a)
before_bar = [count_b[i] * 100 / len(len_b) if i in count_b.keys() else 0 for i in range(1, N + 1)]
after_bar = [count_a[i] * 100 / len(len_a) if i in count_a.keys() else 0 for i in range(1, N + 1)]
ind = np.arange(1, 4)
plt.figure(figsize = (4, 5))
plt.ylabel('Frequency [%]', fontsize = 18)
plt.xlabel('Seconds [s]', fontsize = 18)
width = 0.3
plt.xlim(1 - width, 3 + width * 2)   
plt.ylim(0, 60)
plt.bar(ind, before_bar[:3], width, label = 'Dataset 1', color = '#3498db')
plt.bar(ind + width, after_bar[:3], width, label = 'Dataset 2', color = '#2ecc71')
plt.title('Interaction duration', fontsize = 20)
plt.xticks(ind + width / 2, np.arange(1, 4), fontsize = 18)
plt.yticks([0, 10, 20, 30, 40, 50], fontsize = 18)
leg = plt.legend(prop = {'size': 18}, markerscale = 4)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(N_PATH, 'InteractionNo21.png'), format = 'png', dpi = 300)
plt.savefig(os.path.join(N_PATH, 'InteractionNo21.pdf'), format = 'pdf', dpi = 500)
plt.close()

# %%
# Interaction # 4
statistics_u, pvals = stats.mannwhitneyu(inter_per_class[0], inter_per_class[1], 
    alternative = 'greater')
print ('p-value (Mann-Whitney U test): \t', pvals)
print (f'0: {np.median(inter_per_class[0])}, 1: {np.median(inter_per_class[1])}')
print (f'U: {statistics_u}')

df = pd.DataFrame(columns = ['Number', 'Cluster', 'Color'])
df['Number'] = inter_per_class[0] + inter_per_class[1]
df['Color'] = ['#3498db'] * 294 + ['salmon'] * 84
df['Cluster'] = [0] * 294 + [1] * 84
fig = sns.boxplot(x = 'Cluster', y = 'Number', data = df, palette = ['#3498db', 'salmon'])
fig.set_xlabel('Cluster', fontsize = 15)
fig.set_ylabel('# of interactions', fontsize = 15)
plt.xticks(fontsize = 15)
plt.yticks([0, 10, 20, 30, 40], fontsize = 15)
plt.ylim(-3, 50)
plt.title('Summary per cluster', fontsize = 15)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(N_PATH, 'InteractionNo4.png'), format = 'png', dpi = 300)
plt.savefig(os.path.join(N_PATH, 'InteractionNo4.pdf'), format = 'pdf', dpi = 500)
plt.close()

# %%
