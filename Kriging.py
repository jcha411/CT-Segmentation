import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import skimage
from scipy import stats
from sklearn.linear_model import LinearRegression
import math
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

class KrigingABC(ABC):
    def abc_train(self, slice_idx):
        self.img_height, self.img_width = self.img_full.shape[1:]
        self.num_pixels = self.img_width*self.img_height

        # Extract training slice
        self.img = Image.fromarray(self.img_full[slice_idx])
        self.pixels = np.array(self.img).ravel()

        # Fit Gaussian mixture model on first slice
        self.gm = GaussianMixture(n_components=self.n_clusters, random_state=1).fit(self.pixels.reshape(-1, 1))
        self.pred = self.gm.predict(self.pixels.reshape(-1, 1))

        # Assign segmentation colours to labels
        avg = np.zeros(self.n_clusters)
        counts = np.zeros(self.n_clusters)
        for i in range(self.num_pixels):
            avg[self.pred[i]] += self.pixels[i]
            counts[self.pred[i]] += 1

        for i in range(self.n_clusters):
            avg[i] /= counts[i]

        self.segment_clr = np.empty(self.n_clusters)
        for i in range(self.n_clusters):
            idx = np.argsort(avg)[i]
            self.segment_clr[idx] = i*255/(self.n_clusters-1)

        # Create image histograms
        self.img_hist = np.array(self.img.histogram())
        self.img_chist = np.cumsum(self.img_hist)/self.num_pixels
        self.num_bins = len(self.img_hist)

        # Get uncertain regions
        self.t0s = np.array([])
        self.t1s = np.array([])

        k0 = 0
        k1 = 1
        while k1 < self.n_clusters:
            mu0 = self.gm.means_[k0]
            mu1 = self.gm.means_[k1]
            sigma0 = np.sqrt(self.gm.covariances_[k0])
            sigma1 = np.sqrt(self.gm.covariances_[k1])

            def compute_t(zeta0_):
                z0 = min(mu0 + self.zeta0 * sigma0, mu1 - 0.674 * sigma1)[0, 0]
                z1 = max(mu1 - self.zeta0 * sigma1, mu0 + 0.674 * sigma0)[0, 0]

                t0_ = int(np.round(min(z0, z1)))
                t1_ = int(np.round(max(z0, z1)))

                if t0_ < 0 or t0_ >= self.num_bins or t1_ < 0 or t1_ >= self.num_bins:
                    t_unc = 0
                else:
                    t_unc = (self.img_chist[t1_] - self.img_chist[t0_])

                return t0_, t1_, t_unc

            t0, t1, t_uncertain = compute_t(self.zeta0)

            it = 0
            lr = self.zeta0 * 0.05
            curr_zeta0 = self.zeta0
            while it < self.max_iter_uncertain and (t_uncertain < self.p_uncertain*0.8 or t_uncertain > self.p_uncertain*1.2):
                prev_zeta0 = curr_zeta0 - lr
                next_zeta0 = curr_zeta0 + lr

                prev_t0, prev_t1, prev_t_uncertain = compute_t(prev_zeta0)
                next_t0, next_t1, next_t_uncertain = compute_t(next_zeta0)

                prev_p_uncertain = np.abs(self.p_uncertain - prev_t_uncertain) / self.p_uncertain
                next_p_uncertain = np.abs(self.p_uncertain - next_t_uncertain) / self.p_uncertain

                if (prev_p_uncertain < next_p_uncertain):
                    curr_zeta0 = prev_zeta0
                    t0 = prev_t0
                    t1 = prev_t1
                    t_uncertain = prev_t_uncertain
                else:
                    curr_zeta0 = next_zeta0
                    t0 = next_t0
                    t1 = next_t1
                    t_uncertain = next_t_uncertain

                lr *= 0.8
                it += 1

            if t_uncertain > self.p_uncertain*1.2:
                t0 = 0
                t1 = 0

            self.t0s = np.append(self.t0s, t0)
            self.t1s = np.append(self.t1s, t1)

            k0 += 1
            k1 += 1
        
        print(self.t0s, self.t1s)

        self.abc_infer(slice_idx)

    def abc_infer(self, slice_idx):
        # Extract selected slice
        self.img = Image.fromarray(self.img_full[slice_idx])
        self.pixels = np.array(self.img).ravel()

        # Get prediction and segmentation from GMM
        self.pred = self.gm.predict(self.pixels.reshape(-1, 1))

        self.segment = np.zeros(self.num_pixels)
        for i in range(self.num_pixels):
            self.segment[i] = self.segment_clr[self.pred[i]]
        
        # Uncomment to plot initial Gaussian segmentation
        # out = self.segment.reshape(self.img_height, self.img_width)
        # plt.imsave('01_gaussian_segmentation.png', out, cmap='gray')

        # Label uncertain pixels
        self.uncertain_reg = np.ones(self.num_pixels)
        
        for i in range(self.num_pixels):
            for j in range(self.n_clusters-1):
                if self.pixels[i] > self.t0s[j] and self.pixels[i] <= self.t1s[j]:
                    self.uncertain_reg[i] = 0
        
        # Uncomment to show pixels marked as uncertain
        # out = self.uncertain_reg.reshape(self.img_height, self.img_width)
        # plt.imsave('02_uncertain_pixels.png', out, cmap='gray')

        # Uncomment to show Gaussian curves with uncertain regions highlighted
        # for i in range(self.n_clusters):
        #     mu = self.gm.means_[i]
        #     variance = self.gm.covariances_[i]
        #     sigma = np.sqrt(variance)
        
        #     n = 500
        #     x = np.linspace(0, 256, n)
        #     y = (self.gm.weights_[i]*stats.norm.pdf(x, mu, sigma)).reshape(n)
        #     plt.plot(x, y, color='black')
        
        # plt.bar(range(self.num_bins), self.img_hist/self.num_pixels)
        # for i in range(self.n_clusters-1):
        #     plt.axvspan(self.t0s[i], self.t1s[i], color='red', alpha=0.7, lw=0)
        
        # plt.savefig('03_gaussian_w_uncertain.png')

class TwoDimMultiRegionKriging(KrigingABC):
    def __init__(self, img_full, n_clusters, max_iter_uncertain=10, zeta0=1.96, p_uncertain=0.12, vario_distance=30, krig_win_radius=3):
        self.img_full = img_full
        self.n_clusters = n_clusters
        self.max_iter_uncertain = max_iter_uncertain
        self.zeta0 = zeta0
        self.p_uncertain = p_uncertain

        self.vario_distance = vario_distance
        self.krig_win_radius = krig_win_radius

    def majority_filter(self, image, certain_only):
        out = image.copy()
        
        for y in range(self.img_height):
            for x in range(self.img_width):
                if certain_only and self.uncertain_reg[y*self.img_width + x] == 0:
                    continue
                
                buckets = np.zeros(self.n_clusters)
                for yy in range(y-1, y+2):
                    for xx in range(x-1, x+2):
                        if xx>=0 and xx<self.img_width and yy>=0 and yy<self.img_height:
                            for a in range(self.n_clusters):
                                if self.pixels[yy*self.img_width + x] == self.segment_clr[a]:
                                    buckets[a] += 1
                
                for a in range(self.n_clusters):
                    if buckets[a] >= 5:
                        out[y*self.img_width + x] = self.segment_clr[a]
                        break
        
        return out

    def majority_filter_on_certain(self):
        self.segment = self.majority_filter(self.segment, True).reshape(self.img_height, self.img_width)

        # Uncomment to show majority filter applied to pixels marked as certain
        # plt.imsave('04_majority_filter_on_certain.png', self.segment, cmap='gray')

    def calculate_semivariances(self):
        sv_mu = 0
        sv_count = np.zeros(self.vario_distance)
        sv_pixels = np.array(self.pixels, dtype=np.float64)

        self.sv_sigma2 = 0
        self.sv_range = 0
        self.sv_gamma = np.zeros(self.vario_distance)
        
        for y in range(self.img_height):
            for x in range(self.img_width):
                for d in range(self.vario_distance):
                    id = x + d + 1
                    if id < self.img_width:
                        self.sv_gamma[d] += (sv_pixels[y*self.img_width + x] - sv_pixels[y*self.img_width + id])**2
                        sv_count[d] += 1
        
                    id = y + d + 1
                    if id < self.img_height:
                        self.sv_gamma[d] += (sv_pixels[y*self.img_width + x] - sv_pixels[id*self.img_width + x])**2
                        sv_count[d] += 1
        
                sv_mu += sv_pixels[y*self.img_width + x]
        
        sv_mu /= self.num_pixels
        for d in range(self.vario_distance):
            self.sv_gamma[d] = self.sv_gamma[d] / (sv_count[d] * 2) # or just sv_count[d]*2 ?
        
        for y in range(self.img_height):
            for x in range(self.img_width):
                self.sv_sigma2 += (sv_pixels[y*self.img_width + x] - sv_mu)**2
        
        self.sv_sigma2 /= self.num_pixels
        
        for d in range(self.vario_distance):
            if self.sv_gamma[d] >= self.sv_sigma2:
                self.sv_range = d
                break
        
            self.sv_range = self.vario_distance

    def fit_semivariogram(self):
        assert (self.sv_range > 1), 'ERROR: Unable to identify range value'
        
        data = np.zeros((self.sv_range, 3))
        y = np.zeros(self.sv_range)
        
        for r in range(self.sv_range):
            x = r + 1
            data[r][0] = x
            data[r][1] = x**2
            data[r][2] = x**3
            y[r] = self.sv_gamma[r]
        
        self.sv_clf = LinearRegression()
        self.sv_clf.fit(data, y)
        
        sv_still = self.sv_clf.predict([[self.sv_range, self.sv_range**2, self.sv_range**3]])[0]
        
        pred = self.sv_clf.predict(data)
        regr = np.array([sv_still if self.sv_gamma[r] >= sv_still else pred[i] for i in range(self.sv_range)])

        # Uncomment to show fitted semivariogram
        # plt.close()
        # plt.scatter(data[:, 0], y)
        # plt.plot(data[:, 0], regr)
        # plt.savefig('05_semivariogram.png')

    def solve_kriging(self):
        w_counts = 0
        win_size = (self.krig_win_radius * 2) + 1
        wc = np.array([int(self.krig_win_radius), int(self.krig_win_radius)])
        wp = np.array([0, 0])
        
        for y in range(win_size):
            for x in range(win_size):
                wp[0] = x
                wp[1] = y
                if np.linalg.norm(wp - wc) <= self.krig_win_radius and not (wp[0] == wc[0] and wp[1] == wc[1]):
                    w_counts += 1
        
        self.weights = np.zeros(w_counts)
        idx = 0
        for y in range(win_size):
            for x in range(win_size):
                wp[0] = x
                wp[1] = y
                ed = np.linalg.norm(wp - wc)
        
                if ed <= self.krig_win_radius and not (wp[0] == wc[0] and wp[1] == wc[1]):
                    if ed < self.sv_range:
                        estimated_gamma = self.sv_clf.predict([[ed, ed**2, ed**3]])[0]
                    else:
                        estimated_gamma = self.sv_sigma2
        
                    self.weights[idx] = self.sv_sigma2 - estimated_gamma
                    idx += 1
        
        sum_weights = 0
        for i in range(w_counts):
            sum_weights += self.weights[i]
        
        for i in range(w_counts):
            self.weights[i] /= sum_weights
        
        if self.krig_win_radius > self.sv_range:
            self.krig_win_radius = self.sv_range

    def final_segmentation(self):
        d = math.ceil(self.krig_win_radius)
        c = (0, 0)
        p = (0, 0)
        prob = [0.0, 0.0]
        
        ik = np.zeros(self.pixels.shape[0])
        for y in range(self.img_height):
            for x in range(self.img_width):
                ik[y*self.img_width + x] = self.segment[y*self.img_width + x]
                if self.uncertain_reg[y*self.img_width + x] == 0:
                    for i in range(self.n_clusters-1):
                        if self.pixels[y*self.img_width + x] > self.t0s[i] and self.pixels[y*self.img_width + x] <= self.t1s[i]:
                            t0 = self.t0s[i]
                            t1 = self.t1s[i]
                            clr0 = self.segment_clr[i]
                            clr1 = self.segment_clr[i+1]
                            break

                    c = (x, y)
                    count = 0
                    prob[0] = 0
                    prob[1] = 0
                    
                    for yy in range(y-d, y+d+1):
                        for xx in range(x-d, x+d+1):
                            p = (xx, yy)

                            idx = p[1]*self.img_width + p[0]
                            if idx < 0 or idx >= self.num_pixels:
                                continue
                                
                            ind0 = 0
                            z = self.pixels[idx]
                            if z < t0:
                                ind0 = 1
                            elif z > t1:
                                ind0 = 0
                            else:
                                ind0 = abs(self.img_chist[int(t1)] - self.img_chist[int(z)]) / (self.img_chist[int(t1)] - self.img_chist[int(t0)])
                            
                            ind1 = 0
                            z = self.pixels[idx]
                            if z > t1:
                                ind1 = 1
                            elif z < t0:
                                ind1 = 0
                            else:
                                ind1 = 1.0 - abs(self.img_chist[int(t1)] - self.img_chist[int(z)]) / (self.img_chist[int(t1)] - self.img_chist[int(t0)])
                            
                            if np.linalg.norm(np.array(p)-np.array(c)) <= self.krig_win_radius and not (p[0]==c[0] and p[1]==c[1]):
                                prob[0] += self.weights[count]*ind0
                                prob[1] += self.weights[count]*ind1
                                count += 1
                    
                    if prob[0] > prob[1]:
                        ik[y*self.img_width + x] = clr1
                    else:
                        ik[y*self.img_width + x] = clr0

        out = self.majority_filter(ik, False).reshape(self.img_height, self.img_width)
        return out

    def train(self, slice_idx=0):
        self.abc_train(slice_idx)
        self.majority_filter_on_certain()
        self.calculate_semivariances()
        self.fit_semivariogram()
        self.solve_kriging()

    def infer(self, slice_idx):
        # TODO: Add assertion to check if self.train() has been run
        
        self.abc_infer(slice_idx)
        return self.final_segmentation()

class DeepKriging(KrigingABC):
    def __init__(self, img_full, n_clusters, max_iter_uncertain=10, zeta0=1.96, p_uncertain=0.12, n_epochs=20, batch_size=32):
        self.img_full = img_full
        self.n_clusters = n_clusters
        self.max_iter_uncertain = max_iter_uncertain
        self.zeta0 = zeta0
        self.p_uncertain = p_uncertain

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def encode_coord(self, normalized_lon, normalized_lat):
        num_basis = [10 ** 2, 19 ** 2, 37 ** 2]
        knots_1dx = [np.linspace(0, 1, int(np.sqrt(i))) for i in num_basis]
        knots_1dy = [np.linspace(0, 1, int(np.sqrt(i))) for i in num_basis]

        # Wendland kernel
        basis_size = 0
        phi = np.zeros((normalized_lon.shape[0], sum(num_basis)))
        for res in range(len(num_basis)):
            theta = 1 / np.sqrt(num_basis[res]) * 2.5
            knots_x, knots_y = np.meshgrid(knots_1dx[res], knots_1dy[res])
            knots = np.column_stack((knots_x.flatten(), knots_y.flatten()))
            for i in range(num_basis[res]):
                d = np.linalg.norm(np.vstack((normalized_lon, normalized_lat)).T - knots[i, :], axis=1) / theta
                for j in range(len(d)):
                    if d[j] >= 0 and d[j] <= 1:
                        phi[j, i + basis_size] = (1 - d[j]) ** 6 * (35 * d[j] ** 2 + 18 * d[j] + 3) / 3
                    else:
                        phi[j, i + basis_size] = 0
            basis_size = basis_size + num_basis[res]

        return phi

    def precompute_pixel_encoding(self):
        x, y = np.meshgrid([i for i in range(self.img_width)], [i for i in range(self.img_height)])

        x = x.reshape(-1)
        y = y.reshape(-1)

        normalized_x = x / (self.img_width - 1)
        normalized_y = y / (self.img_height - 1)

        self.phi_obs_full = self.encode_coord(normalized_x, normalized_y)
        self.phi_obs_full = self.phi_obs_full.reshape((self.img_height, self.img_width, -1))

    def prepare_train_test_data(self):
        points = [[] for i in range(self.n_clusters)]

        pred_2d = self.pred.reshape(self.img_height, self.img_width)
        for i in range(self.img_height):
            for j in range(self.img_width):
                if self.uncertain_reg[i * self.img_width + j] == 1:
                    points[pred_2d[i][j]].append([j, i])

        self.phi_obs = []
        z = []
        for i in range(self.n_clusters):
            np.random.shuffle(points[i])
            points[i] = np.array(points[i][:len(points[i]) // 2])

            for p in points[i]:
                self.phi_obs.append(self.phi_obs_full[p[1]][p[0]])
                z.append(i)

        self.phi_obs = np.array(self.phi_obs)
        z_class = np.array(z, dtype=int).reshape(-1, 1)

        z_class = OneHotEncoder(categories=[[i for i in range(self.n_clusters)]], handle_unknown='ignore', sparse_output=False).fit_transform(z_class)

        tensor_x = torch.Tensor(self.phi_obs).to(self.device)
        tensor_y = torch.Tensor(z_class).to(self.device)

        my_dataset = TensorDataset(tensor_x, tensor_y)
        train_size = int(0.8 * len(my_dataset))
        test_size = len(my_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(my_dataset, [train_size, test_size])

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(test_dataset)

    def init_nn_model(self):
        self.net = nn.Sequential(
            nn.Linear(self.phi_obs.shape[1], 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(100),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(100),
            nn.Linear(100, self.n_clusters),
            nn.Softmax(dim=1)
        ).to(self.device)

    def train_nn(self):
        optimizer = torch.optim.Adam(self.net.parameters())
        loss_fn = torch.nn.BCEWithLogitsLoss()

        train_mse = []
        test_mse = []

        for epoch in tqdm(range(self.n_epochs)):
            self.net.train()
            sum_diff = 0
            count = 0
            for data in self.train_dataloader:
                inputs, labels = data
                optimizer.zero_grad()
                outputs = self.net(inputs)

                loss = loss_fn(outputs, labels)
                loss.backward()

                sum_diff += ((outputs - labels) ** 2).float().sum()
                count += outputs.shape[0]

                optimizer.step()

            train_mse.append((sum_diff / count).cpu().detach().numpy())

            sum_diff = 0
            count = 0
            for data in self.test_dataloader:
                self.net.eval()
                with torch.no_grad():
                    inputs, labels = data
                    outputs = self.net(inputs)

                    sum_diff += ((outputs - labels) ** 2).float().sum()
                    count += outputs.shape[0]

            test_mse.append((sum_diff / count).cpu().detach().numpy())
            # print(train_mse[-1], test_mse[-1])

    def final_segmentation(self):
        new_x = []
        new_y = []
        new_phi_obs = []

        for y in range(self.img_height):
            for x in range(self.img_width):
                if self.uncertain_reg[y * self.img_width + x] == 0:
                    new_x.append(x)
                    new_y.append(y)
                    new_phi_obs.append(self.phi_obs_full[y][x])

        new_x = np.array(new_x)
        new_y = np.array(new_y)
        new_phi_obs = np.array(new_phi_obs)

        self.net.eval()
        with torch.no_grad():
            inputs = torch.Tensor(new_phi_obs).to(self.device)
            outputs = self.net(inputs)
            new_labels = (np.round(outputs.cpu().detach().numpy())).astype(int)

        new_labels_single = np.argmax(new_labels, axis=1)

        final_segmentation = self.segment.reshape(self.img_height, self.img_width)
        for i in range(self.n_clusters):
            samples_ix = np.where(new_labels_single == i)
            r = new_y[samples_ix]
            c = new_x[samples_ix]

            for ri, ci in zip(r, c):
                final_segmentation[ri, ci] = self.segment_clr[i]

        return final_segmentation

    def train(self, slice_idx=0):
        self.abc_train(slice_idx)
        self.precompute_pixel_encoding()
        self.prepare_train_test_data()
        self.init_nn_model()
        self.train_nn()
 
    def infer(self, slice_idx):
        # TODO: Add assertion to check if self.train() has been run

        self.abc_infer(slice_idx)
        return self.final_segmentation()