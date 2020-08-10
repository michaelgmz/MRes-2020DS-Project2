# %%
'''----------------------------------------------------------------
Total # of cells identified by the following algorithms
FastER | Mask-RCNN | Imaris
----------------------------------------------------------------'''
def total_cell():
    import pandas as pd
    import pickle

    count_file_before = 'D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 20-160.avi\YellowBlur\cellCounts.txt'
    count_file_after = 'D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 280-400.avi\YellowBlur\cellCounts.txt'
    fastER_count_before = pd.read_table(count_file_before, delimiter = '; ')
    fastER_count_after = pd.read_table(count_file_after, delimiter = '; ')
    print ('---> fastER before: {}\n'.format(sum(fastER_count_before['Position2'])), 
            '---> fastER after: {}'.format(sum(fastER_count_after['Position2'])), 
            sep = '')

    dict_bm = pickle.load(open('D:\Rotation2\VideoFrame\dict_bm.pkl', 'rb'))
    dict_am = pickle.load(open('D:\Rotation2\VideoFrame\dict_am.pkl', 'rb'))
    print ('---> MRCNN before: {}\n'.format(dict_bm['instances'][-1]), 
            '---> MRCNN after: {}'.format(dict_am['instances'][-1]), 
            sep = '')

    overall_before = 'D:\Rotation2\IVM-MRes project #2\IVM machine learning project 2020 (Mingze)\spleen CD11a blocking 19-4-18 before_Statistics\spleen CD11a blocking 19-4-18 before_Overall.csv'
    overall_after = 'D:\Rotation2\IVM-MRes project #2\IVM machine learning project 2020 (Mingze)\spleen CD11a blocking 19-4-18 after_Statistics\spleen CD11a blocking 19-4-18 after_Overall.csv'
    imaris_count_before = pd.read_csv(overall_before, usecols = [0, 1, 3], skiprows = 3)
    imaris_count_after = pd.read_csv(overall_after, usecols = [0, 1, 3], skiprows = 3)
    print ('---> Imaris before: {}\n'.format(int(imaris_count_before['Value'][-1:])), 
            '---> Imaris after: {}'.format(int(imaris_count_after['Value'][-1:])), 
            sep = '')

total_cell()
    
# %%
'''----------------------------------------------------------------
Visualize performance of Imaris and fastER
Colored dots represent different Z-dimension recognized by Imaris
----------------------------------------------------------------'''
def Imaris_fastER(PAR_PATH, IMG_FOLDER, ws_position):
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    if not os.path.exists(os.path.join(PAR_PATH, 'Imaris')):
        os.mkdir(os.path.join(PAR_PATH, 'Imaris'))

    imgs = os.listdir(IMG_FOLDER)
    png_imgs = []
    for img in imgs:
        if img.endswith('.png'):
            png_imgs.append(img)

    for n, img_name in tqdm(enumerate(png_imgs)):
        img = plt.imread(os.path.join(IMG_FOLDER, img_name))
        ws_position_n = ws_position[ws_position['Time'] == n + 1]
            
        plt.imshow(img)
        plt.scatter(ws_position_n['Position X'], ws_position_n['Position Y'], s = 1, c = ws_position_n['Position Z'], 
            cmap = 'gist_rainbow')
        cbar = plt.colorbar(orientation = 'vertical', pad = 0, shrink = 0.85)
        cbar.outline.set_edgecolor('None')
        cbar.ax.tick_params(labelsize = 'xx-small', size = 0)
        plt.axis('off')
        plt.savefig(os.path.join(PAR_PATH, 'Imaris', img_name), format = 'png', dpi = 300, 
            pad_inches = 0, bbox_inches = 'tight')
        plt.close()

    return ws_position

# %%
'''----------------------------------------------------------------
# of cells identified comparison
----------------------------------------------------------------'''
import pickle
import pandas as pd
from numpy import diff
import numpy as np
import scipy.stats as stats
import cv2
import os
from CellOperation import Info
from tqdm import tqdm

def cell_comparison(exp):
    if exp == 'CD11a before':
        faster = pickle.load(open('D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 before both cells video frames 80-140.avi\YellowBlur\dict_bf.pkl', 'rb'))
        mrcnn = pickle.load(open('D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 before both cells video frames 80-140.avi\YellowBlur\dict_bm.pkl', 'rb'))
        dis = pickle.load(open('D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 before both cells video frames 80-140.avi\YellowBlur\discrepancy_b.pkl', 'rb'))
        imaris = pd.ExcelFile('D:\Rotation2\VideoFrame\Exp 19-4-18 CD11a blocking\spleen CD11a blocking 19-4-18 before N frames 80-140.xlsx')
        ws_position = pd.read_excel(imaris, sheet_name = 'Position', header = 1)
        ws_position['Position X'] = ws_position['Position X'] * 1.7105 - 983.6
        ws_position['Position Y'] = ws_position['Position Y'] * (-1.7098) + 1955.9
        IMG_FOLDER = 'D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 before both cells video frames 80-140.avi\YellowBlur\combine'
        PAR_PATH = os.path.split(IMG_FOLDER)[0]
        # Imaris_fastER(PAR_PATH, IMG_FOLDER, ws_position)

    if exp == 'CD11a after':
        faster = pickle.load(open('D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 after both cells frames 340-400.avi\YellowBlur\dict_af.pkl', 'rb'))
        mrcnn = pickle.load(open('D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 after both cells frames 340-400.avi\YellowBlur\dict_am.pkl', 'rb'))
        dis = pickle.load(open('D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 after both cells frames 340-400.avi\YellowBlur\discrepancy_a.pkl', 'rb'))
        imaris = pd.ExcelFile('D:\Rotation2\VideoFrame\Exp 19-4-18 CD11a blocking\spleen CD11a blocking 19-4-18 after N frames 340-400.xlsx')
        ws_position = pd.read_excel(imaris, sheet_name = 'Position', header = 1)
        ws_position['Position X'] = ws_position['Position X'] * 1.7395 - 1007.4
        ws_position['Position Y'] = ws_position['Position Y'] * (-1.7216) + 1964.7
        IMG_FOLDER = 'D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 after both cells frames 340-400.avi\YellowBlur\combine'
        PAR_PATH = os.path.split(IMG_FOLDER)[0]
        # Imaris_fastER(PAR_PATH, IMG_FOLDER, ws_position)

    if exp == 'FTY720 before':
        faster = pickle.load(open('D:\Rotation2\VideoFrame\extract_FTY720 spleen 1h both cells before.avi\YellowBlur\dict_bf.pkl', 'rb'))
        mrcnn = pickle.load(open('D:\Rotation2\VideoFrame\extract_FTY720 spleen 1h both cells before.avi\YellowBlur\dict_bm.pkl', 'rb'))
        dis = pickle.load(open('D:\Rotation2\VideoFrame\extract_FTY720 spleen 1h both cells before.avi\YellowBlur\discrepancy_b.pkl', 'rb'))
        imaris = pd.ExcelFile('D:\Rotation2\VideoFrame\Exp 18-5-18 FTY720\FTY720 spleen 1h N before.xlsx')
        ws_position = pd.read_excel(imaris, sheet_name = 'Position', header = 1)
        ws_position['Position X'] = ws_position['Position X'] * 1.7126 - 994.25
        ws_position['Position Y'] = ws_position['Position Y'] * (-1.7115) + 1052.9
        IMG_FOLDER = 'D:\Rotation2\VideoFrame\extract_FTY720 spleen 1h both cells before.avi\YellowBlur\combine'
        PAR_PATH = os.path.split(IMG_FOLDER)[0]
        # Imaris_fastER(PAR_PATH, IMG_FOLDER, ws_position)

    if exp == 'FTY720 after':
        faster = pickle.load(open('D:\Rotation2\VideoFrame\extract_FTY720 spleen 1h both cells after.avi\YellowBlur\dict_af.pkl', 'rb'))
        mrcnn = pickle.load(open('D:\Rotation2\VideoFrame\extract_FTY720 spleen 1h both cells after.avi\YellowBlur\dict_am.pkl', 'rb'))
        dis = pickle.load(open('D:\Rotation2\VideoFrame\extract_FTY720 spleen 1h both cells after.avi\YellowBlur\discrepancy_a.pkl', 'rb'))
        imaris = pd.ExcelFile('D:\Rotation2\VideoFrame\Exp 18-5-18 FTY720\FTY720 spleen 1h N after.xlsx')
        ws_position = pd.read_excel(imaris, sheet_name = 'Position', header = 1)
        ws_position['Position X'] = ws_position['Position X'] * 1.7257 - 1006.4
        ws_position['Position Y'] = ws_position['Position Y'] * (-1.7318) + 1059.6
        IMG_FOLDER = 'D:\Rotation2\VideoFrame\extract_FTY720 spleen 1h both cells after.avi\YellowBlur\combine'
        PAR_PATH = os.path.split(IMG_FOLDER)[0]
        # Imaris_fastER(PAR_PATH, IMG_FOLDER, ws_position)

    f = list(diff(faster['instances']) / 1)
    f.insert(0, faster['instances'][0])
    print ('fastER')
    print (np.median(f), '\n', stats.iqr(f), sep = '')

    m = list(diff(mrcnn['instances']) / 1)
    m.insert(0, mrcnn['instances'][0])
    print ('MRCNN')
    print (np.median(m), '\n', stats.iqr(m), sep = '')

    instances = []
    for key in dis.keys():
        instances.append(dis[key]['instances'] + m[int(key) - 1])
    print ('FM')
    print (np.median(instances), '\n', stats.iqr(instances), sep = '')

    img_folder = os.path.join(PAR_PATH, 'Colour')
    imgs = os.listdir(img_folder)
    i, matched, i_only, fm_only = [], [], [], []
    i_only_intensity, fm_only_intensity, matched_intensity = [], [], []
    fm_only_volume, matched_volume = [], []
    fm_only_smooth, matched_smooth = [], []

    for time in tqdm(range(1, 61)):
        temp = ws_position[ws_position['Time'] == time]
        i.append(temp.shape[0])

        if time == 1:
            start_m, end_m, start_f, end_f = 0, mrcnn['instances'][0], 0, faster['instances'][0]
        else:
            start_m, end_m, start_f, end_f = mrcnn['instances'][time - 2], mrcnn['instances'][time - 1], \
            faster['instances'][time - 2], faster['instances'][time - 1]
            
        contour_m = mrcnn['contour'][start_m:end_m]
        contour_dis = dis[str(time)]['contour']
        contour_m.extend(contour_dis)

        intensity_m = mrcnn['intensity'][start_m:end_m]
        intensity_dis = dis[str(time)]['intensity']
        intensity_m.extend(intensity_dis)

        volume_m = mrcnn['volume'][start_m:end_m]
        volume_dis = dis[str(time)]['volume']
        volume_m.extend(volume_dis)

        smooth_m = mrcnn['smooth'][start_m:end_m]
        smooth_dis = dis[str(time)]['smooth']
        smooth_m.extend(smooth_dis)

        matched_tp, i_only_tp, fm_only_tp, rows = 0, 0, 0, []
        for index, row in temp.iterrows():
            centroid = (row['Position X'], row['Position Y'])
            for n, contour in enumerate(contour_m):
                if cv2.pointPolygonTest(contour, centroid, False) >= 0:
                    matched_tp += 1

                    matched_intensity.append(intensity_m[n])
                    intensity_m.pop(n)
                    matched_volume.append(volume_m[n] * 0.481 * 0.481)
                    volume_m.pop(n)
                    matched_smooth.append(smooth_m[n])
                    smooth_m.pop(n)

                    contour_m.pop(n)
                    rows.append(index)
                    break

        tempted = temp.drop(rows)
        fm_only_intensity.extend(intensity_m)
        fm_only_volume.extend([v * 0.481 * 0.481 for v in volume_m])
        fm_only_smooth.extend(smooth_m)

        for index, row in tempted.iterrows():
            img = cv2.imread(os.path.join(img_folder, imgs[time - 1]))
            img_grey = img.copy()
            centroid = (int(row['Position X']), int(row['Position Y']))
            img = cv2.circle(img, centroid, int(5 / 0.481), (0, 0, 255), -1)
            img[img[:, :, 1] != 0] = 0

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_grey = cv2.cvtColor(img_grey, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                i_only_intensity.append(result.Intensity(contour, img_grey, (1002, 1002, 3)))

        i_only_tp = temp.shape[0] - matched_tp
        fm_only_tp = end_m - start_m + len(contour_dis) - matched_tp
        matched.append(matched_tp)
        i_only.append(i_only_tp)
        fm_only.append(fm_only_tp)

    print ('Imaris')
    print (np.median(i), '\n', stats.iqr(i))
    print ('Matched')
    print (np.median(matched), '\n', stats.iqr(matched))
    print ('Imaris-Only')
    print (np.median(i_only), '\n', stats.iqr(i_only))
    print ('FM-Only')
    print (np.median(fm_only), '\n', stats.iqr(fm_only), '\n')

    return matched_intensity, i_only_intensity, fm_only_intensity, \
        matched_volume, fm_only_volume, matched_smooth, fm_only_smooth

exps = ['CD11a before', 'CD11a after', 'FTY720 before', 'FTY720 after']
# exps = ['FTY720 before']
result = Info()
intensity_set, volume_set, smooth_set = {}, {}, {}

for exp in exps:
    print (exp)
    intensity_set[exp] = []
    volume_set[exp] = []
    smooth_set[exp] = []

    matched_intensity, i_only_intensity, fm_only_intensity, \
        matched_volume, fm_only_volume, matched_smooth, fm_only_smooth = cell_comparison(exp)

    intensity_set[exp].append([matched_intensity, i_only_intensity, fm_only_intensity])
    intensity_set[exp] = intensity_set[exp][0]
    volume_set[exp].append([matched_volume, fm_only_volume])
    volume_set[exp] = volume_set[exp][0]
    smooth_set[exp].append([matched_smooth, fm_only_smooth])
    smooth_set[exp] = smooth_set[exp][0]

PKL_PATH = 'D:\Rotation2\VideoFrame\FeaturePKL'
pickle.dump(intensity_set, open(os.path.join(PKL_PATH, 'intensity.pkl'), 'wb'))
pickle.dump(volume_set, open(os.path.join(PKL_PATH, 'volume.pkl'), 'wb'))
pickle.dump(smooth_set, open(os.path.join(PKL_PATH, 'smooth.pkl'), 'wb'))

# %%
'''----------------------------------------------------------------
Mcnemar Test
----------------------------------------------------------------'''
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
table1 = np.array([[140, 34], [22, 0]])
table2 = np.array([[108, 15], [29, 0]])
table3 = np.array([[138, 8], [102, 0]])
table4 = np.array([[224, 25], [249, 0]])

result1 = mcnemar(table1, exact = True)
result2 = mcnemar(table2, exact = True)
result3 = mcnemar(table3, exact = True)
result4 = mcnemar(table4, exact = True)

print(f'statistic = {result1.statistic}, p-value = {result1.pvalue}')
print(f'statistic = {result2.statistic}, p-value = {result2.pvalue}')
print(f'statistic = {result3.statistic}, p-value = {result3.pvalue}')
print(f'statistic = {result4.statistic}, p-value = {result4.pvalue}')

# %%
'''----------------------------------------------------------------
Prove FM-Only are real cell (Volume & Smooth)
----------------------------------------------------------------'''
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import scipy.stats as stats
plt.style.use('ggplot')

PKL_PATH = 'D:\Rotation2\VideoFrame\FeaturePKL'
volume_set = pickle.load(open(os.path.join(PKL_PATH, 'volume.pkl'), 'rb'))
smooth_set = pickle.load(open(os.path.join(PKL_PATH, 'smooth.pkl'), 'rb'))

# dataset_index = [1, 2, 5, 6]
# fig, axes = plt.subplots(1, 4, figsize = (24, 4), sharey = True)

for i, dataset in enumerate(volume_set.keys()):
    statistics_u, pvals = stats.kruskal(volume_set[dataset][0], volume_set[dataset][1])
    print ('p-value (Kruskal Wallims test): \t', pvals)
    print (f'U: {statistics_u}')
#     ax1 = sns.distplot(volume_set[dataset][0], ax = axes[i], hist = False, kde = True, 
#         kde_kws = {'shade': True}, color = '#3498db', label = 'Matched') # blue
#     ax2 = sns.distplot(volume_set[dataset][1], ax = axes[i], hist = False, kde = True, 
#         kde_kws = {'shade': True}, label = 'FM-Only') # red

#     axes[i].set_xlim([-5, 120])
#     axes[i].set_yticks([])
#     axes[i].set_xticklabels([int(x) for x in axes[i].get_xticks()], size = 25)
#     # axes[i].set_yticklabels([y for y in axes[i].get_yticks()], size = 25)
#     # axes[i].set_title(f'Dataset {str(dataset_index[i])}', fontsize = 30)
#     leg = axes[i].legend(prop = {'size': 23})

#     for line in leg.get_lines():
#         line.set_linewidth(4)

# axes[0].set_ylabel('Volume', fontsize = 30)
# fig.tight_layout()
# plt.savefig(os.path.join(PKL_PATH, 'VolumeCompare.png'), format = 'png', dpi = 500)
# plt.savefig(os.path.join(PKL_PATH, 'VolumeCompare.svg'), format = 'svg')
# plt.close()

# dataset_index = [1, 2, 5, 6]
# fig, axes = plt.subplots(1, 4, figsize = (24, 4), sharey = True)

for i, dataset in enumerate(smooth_set.keys()):
    statistics_u, pvals = stats.kruskal(smooth_set[dataset][0], smooth_set[dataset][1])
    print ('p-value (Kruskal Wallims test): \t', pvals)
    print (f'U: {statistics_u}')
#     ax1 = sns.distplot(smooth_set[dataset][0], ax = axes[i], hist = False, kde = True, 
#         kde_kws = {'shade': True}, color = '#3498db', label = 'Matched') # blue
#     ax2 = sns.distplot(smooth_set[dataset][1], ax = axes[i], hist = False, kde = True, 
#         kde_kws = {'shade': True}, label = 'FM-Only') # red

#     axes[i].set_xlim([-0.1, 0.9])
#     axes[i].set_xticks([0, 0.3, 0.6, 0.9])
#     axes[i].set_yticks([])
#     axes[i].set_xticklabels([round(x, 1) for x in axes[i].get_xticks()], size = 25)
#     # axes[i].set_yticklabels([round(y, 2) for y in axes[i].get_yticks()], size = 25)
#     leg = axes[i].legend(prop = {'size': 23}, loc = 'upper right')

#     for line in leg.get_lines():
#         line.set_linewidth(4)

# axes[0].set_ylabel('Smooth', fontsize = 30)
# fig.tight_layout()
# plt.savefig(os.path.join(PKL_PATH, 'SmoothCompare.png'), format = 'png', dpi = 500)
# plt.savefig(os.path.join(PKL_PATH, 'SmoothCompare.svg'), format = 'svg')
# plt.close()

# %%
'''----------------------------------------------------------------
Prove Imaris-Only and FM-Only are real cell (Intensity)
----------------------------------------------------------------'''
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
plt.style.use('ggplot')
PKL_PATH = 'D:\Rotation2\VideoFrame\FeaturePKL'
# intensity_set = pickle.load(open(os.path.join(PKL_PATH, 'intensity.pkl'), 'rb'))
# fig, axes = plt.subplots(1, 4, figsize = (24, 5), sharey = True)

dataset_index = [1, 2, 5, 6]
for i, dataset in enumerate(intensity_set.keys()):
    statistics_u, pvals = stats.kruskal(intensity_set[dataset][0], intensity_set[dataset][1], intensity_set[dataset][2])
    print ('p-value (Kruskal Wallims test): \t', pvals)
    print (f'U: {statistics_u}')
#     axes[0].set_ylabel('Intensity', fontsize = 30)
#     ax1 = sns.distplot(intensity_set[dataset][0], ax = axes[i], hist = False, kde = True, 
#         kde_kws = {'shade': True}, color = '#3498db', label = 'Matched') # blue
#     ax2 = sns.distplot(intensity_set[dataset][1], ax = axes[i], hist = False, kde = True, 
#         kde_kws = {'shade': True}, color = '#2ecc71', label = 'Imaris-Only') # green
#     ax3 = sns.distplot(intensity_set[dataset][2], ax = axes[i], hist = False, kde = True, 
#         kde_kws = {'shade': True}, label = 'FM-Only') # red

#     axes[i].set_xlim([-5, 120])
#     axes[i].set_yticks([])
#     axes[i].set_xticklabels([int(x) for x in axes[i].get_xticks()], size = 25)
#     # axes[i].set_yticklabels([y for y in axes[i].get_yticks()], size = 25)
#     axes[i].set_title(f'Dataset {str(dataset_index[i])}', fontsize = 30)
#     leg = axes[i].legend(prop = {'size': 23})

#     for line in leg.get_lines():
#         line.set_linewidth(4)

# fig.tight_layout()
# plt.savefig(os.path.join(PKL_PATH, 'IntensityCompare.png'), format = 'png', dpi = 500)
# plt.savefig(os.path.join(PKL_PATH, 'IntensityCompare.svg'), format = 'svg')

# %%
'''----------------------------------------------------------------
Imaris Position Z distribution
----------------------------------------------------------------'''
def z_distribution(PAR_PATH, ws_position):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    sns.set(font_scale = 1)
    g = sns.distplot(ws_position['Position Z'], kde = False, color = 'blue')
    plt.xlabel(f'Position Z ({chr(956)}m)', fontsize = 15)
    plt.ylabel(f'Cell Frequency', fontsize = 15)
    # g.set_yticks([0, 500, 1000, 2000, 3000, 4000])
    plt.title('Histogram of Z dimension detected by Imaris', fontsize = 15)
    plt.savefig(os.path.join(PAR_PATH, 'ImarisFrequency.png'), format = 'png', dpi = 300, bbox_inches = 'tight')
    plt.savefig(os.path.join(PAR_PATH, 'ImarisFrequency.svg'), format = 'svg', dpi = 300, bbox_inches = 'tight')
    plt.close()

z_distribution()

# %%
'''----------------------------------------------------------------
MRCNN framework image padding for thesis purpose
----------------------------------------------------------------'''
def mrcnn_pad():
    from PIL import Image
    old_img = Image.open(r'D:\Rotation2\VideoFrame\MDS.png')
    old_size = old_img.size
    new_size = (3000, 3000)
    new_img = Image.new('RGB', new_size, color = (255, 255, 255))
    new_img.paste(old_img, (int((new_size[0] - old_size[0]) / 2), int((new_size[1] -old_size[1]) / 2)))
    new_img.save(r'C:\Users\Mingze Gao\Desktop\MRCNNPad.png')

mrcnn_pad()

# %%
'''----------------------------------------------------------------
Illustration how to calculate smoothness of cells
----------------------------------------------------------------'''
def smooth_process(IMG_PATH):
    img1 = cv2.imread(IMG_PATH)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), dtype = np.uint8)
    img1 = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(img1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img2 = img1.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    img2 = cv2.drawContours(img2, contours, -1, (0, 0, 255), 2)

    M = cv2.moments(contours[0])
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    centre = (cx, cy)
    img2 = cv2.circle(img2, centre, 1, (0, 0, 255), 4)

    if not os.path.exists(IMG_PATH[:-4] + 'Red.png'):
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(IMG_PATH[:-4] + 'Red.png', format = 'png', dpi = 300, 
            bbox_inches = 'tight', pad_inches = 0)
        plt.close()

    dst = []
    for coordinate in contours[0]:
        peripheral = coordinate[0]
        dst.append(distance.euclidean(centre, peripheral))
    dst = dst[:141]

    x_dst = np.linspace(1, len(dst), 500)
    f = interp1d(range(1, len(dst) + 1), dst, kind = 'cubic')
    y_dst = f(x_dst)

    dx = 1
    dy = diff(dst) / dx
    newdy = []
    for i, n in enumerate(dy):
        if i % 5 == 0:
            newdy.append(n)

    x_new = np.linspace(1, len(newdy), 500)
    f = interp1d(range(1, len(newdy) + 1), newdy, kind = 'cubic')
    y_smooth = f(x_new)

    return x_dst, y_dst, x_new, y_smooth

def get_img():
    PAR_PATH = 'D:\Rotation2\VideoFrame\MaskExample'
    imgs = os.listdir(PAR_PATH)

    xd = []
    yd = []
    xs = []
    ys = []
    for img in imgs:
        if not img.endswith('Red.png'):
            IMG_PATH = os.path.join(PAR_PATH, img)
            x_dst, y_dst, x_new, y_smooth = smooth_process(IMG_PATH)
            xd.append(x_dst)
            yd.append(y_dst)
            xs.append(x_new)
            ys.append(y_smooth)

    plt.figure(figsize = (20, 3))
    count = 0
    for x, dist in zip(xd, yd):
        x = [item - 1 for item in x]
        plt.plot(x[:int(len(dist))], dist, label = 'Image (%s)' % list(string.ascii_uppercase)[count], 
            linewidth = 3)
        count += 1
    fontsize = 20
    plt.title('Distance to centroid', fontsize = 25)
    plt.legend(loc = 'upper right', prop = {'size': 17})
    plt.ylabel('Distance', fontsize = fontsize)
    plt.xticks(fontsize = fontsize)
    plt.yticks(fontsize = fontsize)
    plt.xlim(0, 140)
    plt.savefig(os.path.join(PAR_PATH, 'dist.png'), format = 'png', dpi = 500, 
        bbox_inches = 'tight', pad_inches = 0)
    plt.savefig(os.path.join(PAR_PATH, 'dist.svg'), format = 'svg', dpi = 500, 
        bbox_inches = 'tight', pad_inches = 0)
    plt.close()

    # plt.figure(figsize = (20, 2))
    # count = 0
    # for x, y in zip(xs, ys):
    #     x = [item - 1 for item in x]
    #     print (round(np.mean(abs(y)), 3))
    #     plt.plot(x, (y), label = 'Image %s' % list(string.ascii_uppercase)[count], 
    #         linewidth = 3)
    #     count += 1

    # plt.title('First derivative of distance to cell centroid', fontsize = 15)
    # plt.legend(loc = 'upper right')
    # plt.yticks(fontsize = 15)
    # plt.savefig(os.path.join(PAR_PATH, 'derivative.png'), format = 'png', dpi = 300, 
    #     bbox_inches = 'tight', pad_inches = 0)
    # plt.close()

import matplotlib.pyplot as plt
import cv2
from scipy.spatial import distance
from numpy import diff
import numpy as np
from scipy.interpolate import interp1d
import string
plt.style.use('ggplot')
get_img()

# %%
