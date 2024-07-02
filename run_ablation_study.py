# Training with percentage

from data import *
from models import *
from train import *
from eval import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import random

# GPU config
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

# seed
set_all_seeds(123)

# Parameters
CSV_file = './CSV/WSI.csv'
folder = './embeddings/PLIP/roi_norm/roi'
perc = 5
percentages = list(range(3, 101, perc))
n_runs = 10
config=1 # 'if 1: Train from Hosp A and adding percentage of Hosp B samples'


# Name to save the results
name_dict = {0: 'Training Hosp 2', 1: 'Training Hosp 1'}
name_dict2 = {0: 'B', 1: 'A'}
save_name = name_dict[config]
plt_name =  save_name + '.pdf'

# Dict to save the metrics. For each institution, we fill two list (i.e., avg and std results).
metrics = {}
metrics['AE'] = {'A':np.zeros((len(percentages),n_runs)), 'B': np.zeros((len(percentages),n_runs))}
metrics['DAUD'] = {'A':np.zeros((len(percentages),n_runs)), 'B': np.zeros((len(percentages),n_runs))}

# Data
data = data_partition(CSV_file, config)

# Test set
X_test_A = get_emb_av(data.A_test['values'], folder)
X_test_B = get_emb_av(data.B_test['values'], folder)

# Train set
train_dataset = pd.DataFrame({'WSI': data.train_config['values'], 'outlier':  data.train_config['labels']})    
n_original_train = len(train_dataset)

for run_ix in range(n_runs):
    for perc_ix, percentage in enumerate(percentages):         
        # Samples to add
        num_samples_to_add = int(len(data.train_config['perc_values']) * (percentage / 100))
        pairs = list(zip(data.train_config['perc_values'], data.train_config['perc_labels']))  # make pairs out of the two lists
        pairs = random.sample(pairs, num_samples_to_add)  # pick random pairs
        subset, labels_subset = zip(*pairs)
        subset = data.train_config['perc_values'][:num_samples_to_add]
        labels_subset = data.train_config['perc_labels'][:num_samples_to_add]
        current_train_dataset = pd.concat([train_dataset, pd.DataFrame({'WSI': subset, 'outlier': labels_subset})])
        X_train = get_emb_av(current_train_dataset['WSI'].values, folder)
        y_train = current_train_dataset['outlier'].values
        print('Training size: ', len(y_train))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)


        dom = torch.cat((torch.ones(n_original_train), torch.zeros(len(X_train)-n_original_train)), axis=0)
        X_train = loader(X_train, dom, DEVICE, 'train')
        
        # Test
        X_test_A = get_emb_av(data.A_test['values'], folder)
        X_test_B = get_emb_av(data.B_test['values'], folder)
        X_test_A, X_test_B = scaler.transform(X_test_A),scaler.transform(X_test_B)
        X_test_A, X_test_B = loader(X_test_A, data.dom_test_A, DEVICE, 'test'), loader(X_test_B, data.dom_test_B, DEVICE, 'test')  
        
        for model_name in ['AE', 'DAUD']:
            # Model
            model = select_model(model_name)
            model = model.to(DEVICE)

            trained_model = training_loop(model, X_train, model_name, DEVICE)
            
            # Compute metrics
            scores_A = evaluate(trained_model, model_name, X_test_A, DEVICE)
            scores_B = evaluate(trained_model, model_name, X_test_B, DEVICE)

            auc_A = calculate_metrics(data.A_test['labels'], scores_A)
            auc_B = calculate_metrics(data.B_test['labels'], scores_B)

            # Add metrics
            metrics[model_name]['A'][perc_ix, run_ix] = auc_A
            metrics[model_name]['B'][perc_ix, run_ix] = auc_B



final_metrics = {'AE': {}, 'DAUD':{}}
for model_name in ['AE', 'DAUD']:
    final_metrics[model_name]['A'] =  metrics[model_name]['A'].mean(1), metrics[model_name]['A'].std(1)
    final_metrics[model_name]['B'] =  metrics[model_name]['B'].mean(1), metrics[model_name]['B'].std(1)


save_data(percentages, final_metrics, save_name)

plot_data_(plt_name, save_name, percentages, final_metrics, name_dict2[config])
