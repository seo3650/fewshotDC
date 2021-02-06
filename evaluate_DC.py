import torch
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

import FSLTask

def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    S_d = np.empty(0)
    for mean in base_means:
        mean = torch.tensor(mean)
        S_d = np.append(S_d, np.linalg.norm(mean - query))
    S_n = S_d.argsort()[:k]

    calibrated_mean = np.concatenate([base_means[S_n], query[np.newaxis, :]])
    calibrated_mean = np.mean(calibrated_mean, axis=0)
    calibrated_cov = np.mean(base_cov[S_n], axis=0) + alpha

    return calibrated_mean, calibrated_cov

if __name__ == '__main__':
    # ---- data loading
    dataset = 'CUB'
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 100
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    # Load Few-Shot Learning tasks
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)
    # ---- Base class statistics
    base_means, base_cov = [], []  
    base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk" % dataset

    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])# (items # in same label, 640)
            mean = feature.mean(axis=0)
            cov = np.cov(feature.T)
            
            base_means.append(mean)
            base_cov.append(cov)
    
    base_means = np.array(base_means)
    base_cov = np.array(base_cov)
    # ---- classification for each task
    acc_list = []
    for i in tqdm(range(n_runs)):
        ndata = ndatas[i] # (n_samples, dimension:640)
        label = labels[i] # (n_samples)

        # Tukey's Ladder of Powers Transformation
        lam = 0.5
        ndata = np.power(ndata, lam)

        # Distribution calibration 
        support_data = ndata[:n_lsamples]
        support_label = label[:n_lsamples]
        query_data = ndata[n_lsamples:]
        query_label = label[n_lsamples:]

        n_generate = 2000
        generate_data = []
        generate_label = []
        for sample, label in zip(support_data, support_label):
            calibrated_mean, calibrated_cov = distribution_calibration(sample, base_means, base_cov, 2)
            generate_data.extend(np.random.multivariate_normal(calibrated_mean, calibrated_cov, size=n_generate))
            generate_label.extend([label] * n_generate)
            
        support_data = np.append(generate_data, support_data, axis=0)
        support_label = np.append(generate_label, support_label)
        
        # Classifier
        classifier = LogisticRegression(max_iter=5000).fit(support_data, support_label)
        predicted = classifier.predict(query_data)
        acc = np.mean(predicted == np.array(query_label))
        print(acc)
        acc_list.append(acc)
    print('%s %d way %d shot  ACC : %f'%(dataset, n_ways, n_shot, float(np.mean(acc_list))))