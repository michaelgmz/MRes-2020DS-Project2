# %%
'''----------------------------------------------------------------
Relevant imports and pre-defined parameters
----------------------------------------------------------------'''
import pickle
import os
import sys
from scipy.spatial import distance as dist
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from collections import Counter
import statsmodels.api as sm
import cv2
from tqdm import tqdm
import matplotlib
from sklearn.cluster import DBSCAN
from scipy.stats import zscore
from sklearn.metrics import silhouette_score
plt.style.use('ggplot')
sys.path.append('..')
scale = 0.481
# VIDEO_PATH = 'D:\Rotation2\VideoFrame\SecondCD11a'
VIDEO_PATH = 'D:\Rotation2\VideoFrame'
time_per_frame = 1

# %%
'''----------------------------------------------------------------
self.time_since_update, int, # of frames since last hit, 0 means currently in frame
self.id, set, may contain multiple ids
self.insert, set, frame # when the corresponding multiple id was inserted
self.history_pre, list, bbox prediction history
self.history_upd, dict, {frame # : [x, y, s, r]}, bbox center update history
self.hits, int, # of frames the cell has been hit
self.pat, dict, {frame # : [intensity, volume, smooth, circumference, centre, 
                (startX, startY, endX, endY), velocity, acceleration, displacement]}, 
                patterns of a tracking cell
self.cont, np.array, {frame # : contour}, contour of the detected cell
----------------------------------------------------------------'''
ncell_before = pickle.load(open(os.path.join(VIDEO_PATH, 'neutrophilbefore.pkl'), 'rb'))
ncell_after = pickle.load(open(os.path.join(VIDEO_PATH, 'neutrophilafter.pkl'), 'rb'))

# %%
'''----------------------------------------------------------------
Add Velocity, Acceleration & Displacement features
----------------------------------------------------------------'''
def add_feature(dataset):
    '''
    velocity -> self.pat[frame #][6]
    acceleration -> self.pat[frame #][7]
    displacement -> self.pat[frame #][8]
    '''
    for cell in dataset:
        frame_total = list(cell.pat.keys())
        frame_appear = min(frame_total)

        for n, frame in enumerate(frame_total):
            if frame == frame_appear:
                # cell.pat[frame].extend([(0, 0), (0, 0), (0, 0)])
                cell.pat[frame].extend([0, 0, 0])

            else:
                frame_start = frame_total[n - 1]
                frame_end = frame_total[n]
                time_interval = (frame_end - frame_start) * time_per_frame

                delta = dist.euclidean(cell.history_upd[frame_end][0:2], cell.history_upd[frame_start][0:2])
                velocity = delta / time_interval
                acceleration = (velocity - cell.pat[frame_start][6]) / time_interval
                displacement = delta / (frame_end - frame_start)
                cell.pat[frame_end].extend([velocity, acceleration, displacement])

add_feature(ncell_before)
add_feature(ncell_after)

# %%
'''----------------------------------------------------------------
Saved modifed neutrophil infos
----------------------------------------------------------------'''
pickle.dump(ncell_before, open(os.path.join(VIDEO_PATH, 'ncellbefore.pkl'), 'wb'))
pickle.dump(ncell_after, open(os.path.join(VIDEO_PATH, 'ncellafter.pkl'), 'wb'))

# %%
'''----------------------------------------------------------------
Generate training data (Before After & Both)
----------------------------------------------------------------'''
def generate_train_set(dataset):
    data_one_id = []
    data_all_id = []
    id_one = []
    id_all = []

    for cell in dataset:
        if cell.hits >= 3:
            frame_total = list(cell.pat.keys())

            smooth = []
            velocity = []
            acceleration = []
            intensity = []
            volume = []
            circumference = []
            displacement = []            

            for n, frame in enumerate(frame_total):
                intensity.append(cell.pat[frame][0])
                volume.append(cell.pat[frame][1])
                smooth.append(cell.pat[frame][2])
                circumference.append(cell.pat[frame][3])
                velocity.append(cell.pat[frame][6])
                acceleration.append(cell.pat[frame][7])
                displacement.append(cell.pat[frame][8])

            if len(cell.id) == 1:
                id_one.append(list(cell.id)[0])
                data_one_id.append([np.mean(intensity), np.mean(volume) * scale * scale, 
                    np.mean(smooth), np.mean(circumference) * scale, np.mean(velocity) * scale,
                    np.mean([abs(number) for number in acceleration]) * scale, 
                    np.mean(displacement) * scale])
            
            id_all.append(cell.id)
            data_all_id.append([np.mean(intensity), np.mean(volume) * scale * scale, 
                np.mean(smooth), np.mean(circumference) * scale, np.mean(velocity) * scale,
                np.mean([abs(number) for number in acceleration]) * scale, 
                np.mean(displacement) * scale])

    X_one_id = np.array(data_one_id)
    X_all_id = np.array(data_all_id)
    print (f'---> Shape of dataset: {np.shape(X_one_id)}')
    print (f'---> Shape of dataset: {np.shape(X_all_id)}')

    X_one_id = pd.DataFrame(data = X_one_id, columns = ['Intensity', f'Volume ({chr(956)}m\u00B2)', 'Smooth', 
        f'Circumference ({chr(956)}m)', f'Velocity ({chr(956)}m/s)', f'Acceleration ({chr(956)}m/s\u00B2)', 
        f'Displacement ({chr(956)}m/s)'])
    X_all_id = pd.DataFrame(data = X_all_id, columns = ['Intensity', f'Volume ({chr(956)}m\u00B2)', 'Smooth', 
        f'Circumference ({chr(956)}m)', f'Velocity ({chr(956)}m/s)', f'Acceleration ({chr(956)}m/s\u00B2)', 
        f'Displacement ({chr(956)}m/s)'])
    X_one_id = X_one_id.round(3)
    X_all_id = X_all_id.round(3)

    return X_one_id, X_all_id, id_one, id_all

train_set_before, all_data_before, id_one_before, id_all_before = generate_train_set(ncell_before)
train_set_after, all_data_after, id_one_after, id_all_after = generate_train_set(ncell_after)

train_set_total = pd.concat([train_set_before, train_set_after])
index = ['Before'] * (len(train_set_before)) + ['After'] * (len(train_set_after))
train_set_total['Treatment'] = index

all_data_total = pd.concat([all_data_before, all_data_after])
index = ['Before'] * (len(all_data_before)) + ['After'] * (len(all_data_after))
all_data_total['Treatment'] = index

X_one_id = train_set_total
id_one_before.extend(id_one_after)
X_one_id['ID'] = id_one_before
X_all_id = all_data_total
id_all_before.extend(id_all_after)
X_all_id['ID'] = id_all_before

# %%
train_set_total = pd.concat([one, two])

# %%
'''----------------------------------------------------------------
Pairwise plot using vital features (Scatter & KDE)
----------------------------------------------------------------'''
used_features = train_set_total[['Intensity', f'Volume ({chr(956)}m\u00B2)', 'Smooth', f'Velocity ({chr(956)}m/s)', 'Treatment']]
plot_kws = {'s': 15}
sns.set(font_scale = 1.2)

g = sns.pairplot(used_features, hue = 'Treatment', palette = 'bright', markers = '+', plot_kws = plot_kws)
plt.subplots_adjust(top = 0.95)
g.fig.suptitle('Pairwise Relationship between Selected Features', fontsize = 18)
plt.savefig(os.path.join(VIDEO_PATH, 'Relationship.png'), format = 'png', dpi = 500)
plt.savefig(os.path.join(VIDEO_PATH, 'Relationship.svg'), format = 'svg')
plt.close()

# %%
'''----------------------------------------------------------------
Linear Regression
----------------------------------------------------------------'''
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
v = train_set_before[[f'Velocity ({chr(956)}m/s)']]
a = train_set_before[[f'Acceleration ({chr(956)}m/s\u00B2)']]

import statsmodels.api as sm
x = sm.add_constant(v)
est = sm.OLS(a, x)
est2 = est.fit()
print (est2.summary())

# %%
'''----------------------------------------------------------------
Pairwise plot using motion features (Scatter & KDE)
----------------------------------------------------------------'''
used_features = X_one_id[[f'Velocity ({chr(956)}m/s)', f'Acceleration ({chr(956)}m/s\u00B2)', f'Volume ({chr(956)}m\u00B2)', 
    f'Circumference ({chr(956)}m)', 'Treatment']]
plot_kws = {'s': 15}
sns.set(font_scale = 1.2)

g = sns.pairplot(used_features, hue = 'Treatment', palette = 'bright', markers = '+', plot_kws = plot_kws)
plt.subplots_adjust(top = 0.95)
g.fig.suptitle('Pairwise Relationship between Selected Features', fontsize = 18)
plt.savefig(os.path.join(VIDEO_PATH, 'MotionRela.png'), format = 'png', dpi = 500)
plt.savefig(os.path.join(VIDEO_PATH, 'MotionRela.svg'), format = 'svg')
plt.close()

# %%
'''----------------------------------------------------------------
Q-Q Plot to test data normality distribution 
----------------------------------------------------------------'''
col_names = [f'Circumference ({chr(956)}m)', f'Volume ({chr(956)}m\u00B2)', 'Smooth', 
    f'Velocity ({chr(956)}m/s)', f'Acceleration ({chr(956)}m/s\u00B2)']
fig = plt.figure(figsize = (20, 8))

for n, col in enumerate(col_names):
    ax_before = fig.add_subplot(2, 5, 1 + n)
    sm.qqplot(train_set_before[col], line = '45', fit = True, ax = ax_before)
    ax_before.set_title(col + ' (Before)', fontsize = 18)

    ax_after = fig.add_subplot(2, 5, 6 + n)
    sm.qqplot(train_set_after[col], line = '45', fit = True, ax = ax_after)
    ax_after.set_title(col + ' (After)', fontsize = 18)

fig.suptitle('Q-Q Plot for Normality Test (Before & After Treatment)', fontsize = 20)
fig.tight_layout()
fig.subplots_adjust(top = 0.91)
plt.savefig(os.path.join(VIDEO_PATH, 'QQ-Plot.png'), format = 'png', dpi = 500)
plt.savefig(os.path.join(VIDEO_PATH, 'QQ-Plot.svg'), format = 'svg')
plt.close()

# %%
'''----------------------------------------------------------------
Mean feature comparison bar plot with errorbar (Non-normal, Mann-Whitney U)
----------------------------------------------------------------'''
def single_std(array):
    return np.std(array) * 1
def ci(array):
    return stats.iqr(array)

selected_data = train_set_total
comparison = selected_data.groupby('Treatment').agg([np.median, single_std, ci], axis = 'columns')
after, before = comparison.iloc[0].copy(), comparison.iloc[1].copy()
comparison.iloc[0], comparison.iloc[1] = before, after
col_names = [f'Circumference ({chr(956)}m)', f'Volume ({chr(956)}m\u00B2)', 'Smooth', 
    f'Velocity ({chr(956)}m/s)', f'Acceleration ({chr(956)}m/s\u00B2)']

fig, axes = plt.subplots(1, 5)
for n, col in enumerate(col_names):
    statistics_b, pvals = stats.shapiro(train_set_before[col])
    print (col, f'p-value (Shapiro Before): {pvals}, S: {statistics_b}')
    print (f'df: {len(train_set_before[col])}')

    statistics_a, pvals = stats.shapiro(train_set_after[col])
    print (col, f'p-value (Shapiro After): {pvals}, S: {statistics_a}')
    print (f'df: {len(train_set_after[col])}')

    statistics_u, pvals = stats.mannwhitneyu(train_set_before[col], train_set_after[col])
    print (col, 'p-value (Mann-Whitney U test): \t', pvals)
    print (f'Before: {np.median(train_set_before[col])}, After: {np.median(train_set_after[col])}')
    print (f'U: {statistics_u}')
    
    df = comparison[col]
    ax = df.plot(kind = 'bar', y = 'median', legend = False, title = col, 
        color = ['steelblue', 'seagreen'], width = 0.6, ax = axes[n], zorder = 2, 
        figsize = (16, 5))
    ax.title.set_size(17)
    ax.set_xlabel("")
    ax.set_xticklabels(['Before', 'After'], rotation = 0)
    ax.tick_params(labelsize = 16)
    ax.errorbar(df.index, df['median'], yerr = df['ci'] / 2, capthick = 1, zorder = 1, 
        linewidth = 1.5, color = "black", alpha = 1, capsize = 5, fmt = 'none')

fig.suptitle('Feature Comparison for Before & After treatment', fontsize = 20)
plt.tight_layout()
fig.subplots_adjust(top = 0.85)
plt.savefig(os.path.join(VIDEO_PATH, 'FeatureCompare.png'), format = 'png', dpi = 500)
plt.savefig(os.path.join(VIDEO_PATH, 'FeatureCompare.svg'), format = 'svg')
plt.close()

# %%
'''----------------------------------------------------------------
Build GMM model use only BEFORE data and apply to AFTER data
----------------------------------------------------------------'''
# n_components = np.arange(1, 8)
# features_before = train_set_before[['Intensity', f'Volume ({chr(956)}m\u00B2)', 'Smooth', 
#     f'Velocity ({chr(956)}m/s)', f'Acceleration ({chr(956)}m/s\u00B2)']]
# features_before = (features_before - features_before.min()) / (features_before.max() - features_before.min())
# models = [GaussianMixture(n, covariance_type = 'full', random_state = 1).fit(features_before) 
#     for n in n_components]
# bic = [m.bic(features_before) for m in models]
# plt.plot(n_components, [m.bic(features_before) for m in models], label = 'BIC', 
#     marker = 'o', zorder = 2)
# plt.plot(n_components, [m.aic(features_before) for m in models], label = 'AIC', 
#     marker = 'o', zorder = 2)

# n_components = np.arange(2, 9)
# silhouette = []
# labels = []
# df = train_set_total[[f'Volume ({chr(956)}m\u00B2)', 'Smooth', f'Velocity ({chr(956)}m/s)']]
# normalized_df = df.apply(zscore)
# for k in n_components:
#     kmeans = KMeans(init = 'k-means++', n_clusters = k, n_init = 10, max_iter = 300)
#     kmeans.fit(normalized_df)
#     labels.append(kmeans.labels_)
#     silhouette.append(silhouette_score(normalized_df, kmeans.labels_, metric = 'euclidean'))
# plt.plot(n_components, silhouette, marker = 'o', zorder = 2)
# plt.axvline(x = silhouette.index(max(silhouette)) + 2, linestyle = 'dashed', color = 'black', 
#     zorder = 1)
# plt.legend(loc = 'upper right')

import scipy.cluster.hierarchy as shc
shc.set_link_color_palette(['salmon', '#3498db', '#2ecc71'])
plt.figure()
plt.title('Hierarchical Clustering Dendograms', fontsize = 15)
df = train_set_total[[f'Acceleration ({chr(956)}m/s\u00B2)', 'Smooth', f'Velocity ({chr(956)}m/s)', 'Intensity']]
df = df.apply(zscore)
dend = shc.dendrogram(shc.linkage(df, method = 'ward', optimal_ordering = True), 
    no_labels = True)

plt.axhline(y = 30, linestyle = 'dashed', color = 'black', zorder = 1)
plt.xlabel('Individual cells', fontsize = 14)
plt.ylabel('Dissimilarity between clusters', fontsize = 14)
plt.tick_params(labelsize = 10)

plt.tight_layout()
plt.savefig(os.path.join(VIDEO_PATH, 'Hierarchical.png'), format = 'png', dpi = 500)
plt.savefig(os.path.join(VIDEO_PATH, 'Hierarchical.pdf'), format = 'pdf', dpi = 500)
plt.close()

# %%
# gmm = models[bic.index(min(bic))]
# print ('Converged:', gmm.converged_)
# features_after = train_set_after[['Intensity', f'Volume ({chr(956)}m\u00B2)', 'Smooth', 
#     f'Velocity ({chr(956)}m/s)', f'Acceleration ({chr(956)}m/s\u00B2)']]
# df_after = (features_after - features_after.min()) / (features_after.max() - features_after.min())
# labels = np.concatenate([labels_before, labels_after])
# train_set_total['Labels'] = labels
# counter_before = Counter(labels_before)
# percentage_before = {i:round(counter_before[i] / len(labels_before) * 100, 2) for i in counter_before}
# print ([(i, str(percentage_before[i]) + '%', counter_before[i]) for i in counter_before])

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
labels = cluster.fit_predict(df)

index_list = [239, 225 + 239, 197 + 225 + 239, 181 + 197 + 225 + 239]
for i in range(len(index_list)):
    labels_after = labels[:index_list[i]] if i == 0 else labels[index_list[i - 1]: index_list[i]]
    counter_after = Counter(labels_after)
    percentage_after = {i:round(counter_after[i] / len(labels_after) * 100, 2) for i in counter_after}
    print ([(i, str(percentage_after[i]) + '%', counter_after[i]) for i in counter_after])

# %%
colorsList = ['#3498db', 'salmon']
CustomCmap = matplotlib.colors.ListedColormap(colorsList)
scatter = plt.scatter(train_set_before['Smooth'], train_set_before[f'Velocity ({chr(956)}m/s)'], 
    c = labels[225 + 239:197 + 225 + 239], cmap = CustomCmap, s = 15)
plt.xlabel('Smooth', fontsize = 14)
plt.ylabel(f'Velocity ({chr(956)}m/s)', fontsize = 14)
plt.xlim([0.3, 0.6])
plt.ylim([0, 3.5])
plt.legend(handles = scatter.legend_elements()[0], labels = [0, 1], loc = 'upper right')
plt.title(f'Putative {len(set(labels))} types of neutrophils', fontsize = 15)
plt.tight_layout()
plt.savefig(os.path.join(VIDEO_PATH, 'VolumeVelocity.png'), format = 'png', dpi = 500)
plt.savefig(os.path.join(VIDEO_PATH, 'SmoothVelocity.pdf'), format = 'pdf', dpi = 500)
plt.close()

# %%
'''----------------------------------------------------------------
Chi-square test of independence of variables in a contingency table (raw cell number)
Stacked bar plot (Percentage accounted)
----------------------------------------------------------------'''
# obs = np.empty((2, len(set(labels))))
# for i in range(len(set(labels))):
#     obs[0][i] = counter_before[i]
#     obs[1][i] = counter_after[i]
obs = np.array([[148, 49], [146 * 197 / 181, 35 * 197 / 181]])
chi2, p, dof, expected = stats.chi2_contingency(obs)
print (f'Sample size: {np.sum(obs)}')
print (f'Chi-square： {chi2}, p value: {p}, dof: {dof}, expected: {expected}')

cols = ['Before Treatment', 'After treatment']
percentage = pd.DataFrame(data = obs.T, columns = cols)
percentage[cols] = round(percentage[cols].div(percentage[cols].sum(axis = 0), axis = 1).multiply(100), 2)
percentage = pd.DataFrame(data = percentage.values.T, columns = percentage.index, index = percentage.columns)

colorsList = ['#3498db', 'salmon']
CustomCmap = matplotlib.colors.ListedColormap(colorsList)
ax = percentage.plot(kind = 'bar', stacked = True, title = 'Percentage of cell sub-populations (%)', width = 0.3, 
    alpha = 0.7, cmap = CustomCmap, figsize = (5, 5))
ax.title.set_size(15)
ax.set_xlabel("")
ax.legend([0, 1])
ax.set_yticks([0.00, 75.13, 80.66, 100])
ax.set_xticklabels(['Before Treatment', 'After Treatment'], rotation = 0)
ax.tick_params(labelsize = 12)

plt.tight_layout()
plt.savefig(os.path.join(VIDEO_PATH, 'Percentage2.png'), format = 'png', dpi = 500)
plt.savefig(os.path.join(VIDEO_PATH, 'Percentage2.pdf'), format = 'pdf', dpi = 500)
plt.close()

# %%
'''----------------------------------------------------------------
Multi-dimensional scaling manifold learning
----------------------------------------------------------------'''
# used_features = train_set_total[['Intensity', f'Volume ({chr(956)}m\u00B2)', 'Smooth', f'Velocity ({chr(956)}m/s)', 
#     f'Acceleration ({chr(956)}m/s\u00B2)']]
# used_features = used_features.apply(stats.zscore)

distances = metrics.pairwise_distances(df)
mds = MDS(n_components = 2, metric = True, dissimilarity = 'precomputed', n_init = 15, random_state = 1, 
    eps = 1e-5, max_iter = 400)
res = mds.fit_transform(distances)
scatter = plt.scatter(res[:, 0], res[:, 1], c = labels, cmap = CustomCmap, s = 10)

plt.title('MDS Dimensionality Reduction', fontsize = 15)

plt.legend(handles = scatter.legend_elements()[0], labels = [0, 1], loc = 'best')
plt.tick_params(axis = 'both', bottom = False, left = False, labelbottom = False, labelleft = False)
plt.xlabel('MDS1', fontsize = 14)
plt.ylabel('MDS2', fontsize = 14)
plt.tight_layout()
plt.savefig(os.path.join(VIDEO_PATH, 'MDS.png'), format = 'png', dpi = 500)
plt.savefig(os.path.join(VIDEO_PATH, 'MDS.pdf'), format = 'pdf', dpi = 500)
plt.close()

# %%
'''----------------------------------------------------------------
Visualise cells with only one label and cells with multiple labels
----------------------------------------------------------------'''
# def Mark_Cells(frame_dir_path, ncell_stats):
#     frames = os.listdir(frame_dir_path)
#     imgs_singleid = []
#     imgs_multipleid = []

#     for frame in tqdm(frames):
#         img_path = os.path.join(frame_dir_path, frame)
#         imgs_singleid.append(cv2.imread(img_path))
#         imgs_multipleid.append(cv2.imread(img_path))

#     for cell in ncell_stats:
#         if cell.hits >= 3:
#             keys = cell.pat.keys()

#             if len(cell.id) == 1:
#                 for key in keys:
#                     bbox = cell.pat[key][5]
#                     cv2.rectangle(imgs_singleid[key - 1], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
#                     cv2.putText(imgs_singleid[key - 1], str(list(cell.id)[0]), (int((bbox[0] + bbox[2]) / 2), 
#                         int((bbox[1] + bbox[3]) / 2)), cv2.FONT_HERSHEY_PLAIN, 0.75, (255, 255, 255), 1)
            
#             else:
#                 for key in keys:
#                     bbox = cell.pat[key][5]
#                     cv2.rectangle(imgs_multipleid[key - 1], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)

#     return imgs_singleid, imgs_multipleid

# def Visualise_Cells(frame_dir_path, imgs, idtype):
#     file_path = os.path.join(os.path.split(frame_dir_path)[0], idtype + '.avi')
#     fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#     size = (1002, 1002)
#     video = cv2.VideoWriter(file_path, fourcc, 5, size)

#     for img in imgs:
#         video.write(img)
#     video.release()
    
# FRAME_DIR_BEFORE = 'D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 20-160.avi\YellowBlur\Colour'
# FRAME_DIR_AFTER = 'D:\Rotation2\VideoFrame\extract_spleen CD11a blocking 19-4-18 280-400.avi\YellowBlur\Colour'

# imgs_singleid, imgs_multipleid = Mark_Cells(FRAME_DIR_BEFORE, ncell_before)
# Visualise_Cells(FRAME_DIR_BEFORE, imgs_singleid, 'Single')
# Visualise_Cells(FRAME_DIR_BEFORE, imgs_multipleid, 'Multiple')

# imgs_singleid, imgs_multipleid = Mark_Cells(FRAME_DIR_AFTER, ncell_after)
# Visualise_Cells(FRAME_DIR_AFTER, imgs_singleid, 'Single')
# Visualise_Cells(FRAME_DIR_AFTER, imgs_multipleid, 'Multiple')