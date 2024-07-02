#pyod for ood

# Supervised OoD

# spitz ood

from data import *
from models import *
from train import *
from eval import *
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import TensorDataset, DataLoader
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random

set_all_seeds(123)

# GPU config
cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

# Parameters
CSV_file = './CSV/WSI.csv'
folder = './embeddings/PLIP/roi_norm/roi'
n_runs = 10
config = 0

model_names = ['DAUD', 'ECOD', 'DeepSVDD', 'AnoGAN', 'VAE', 'beta-VAE']
metrics_bin = {}
for model_name in model_names:
    metrics_bin[model_name] = []


# Data
data = data_partition_A_B_ood(CSV_file, config, unsupervised=False)

# Test set
X_test_A = get_emb_av(data.A_test['values'], folder)
X_test_B = get_emb_av(data.B_test['values'], folder)
X_test_A, X_test_B = np.array(X_test_A), np.array(X_test_B)
iid_good = np.concatenate((X_test_A[data.A_test['labels']==0], X_test_B[data.B_test['labels']==0]))
n_A, n_B = len(X_test_A[data.A_test['labels']==0]), len(X_test_B[data.B_test['labels']==0])

ood = get_all_emb('embeddings/PLIP/spitzoides_norm_PLIP')
ood= np.stack(ood, axis=0)
external_test = np.concatenate((iid_good, ood))
labels_external = np.concatenate((np.zeros(len(iid_good)), np.ones(len(ood))))
dom_test = torch.cat((torch.ones(n_A),torch.zeros(n_B), -1*torch.ones(len(ood))))

# Train set
train_dataset = pd.DataFrame({'WSI': data.A_train['values'], 'outlier':  data.A_train['labels']})    
n_original_train = len(train_dataset)

for run_ix in range(n_runs):     
    # Samples to add
    

    for model_name in model_names:
        # Model
        print(model_name)
        # Model
        model = select_model(model_name)

        current_train_dataset = pd.concat([train_dataset, pd.DataFrame({'WSI': data.B_train['values'], 'outlier': data.B_train['labels']})])
        X_train = get_emb_av(current_train_dataset['WSI'].values, folder)
        y_train = current_train_dataset['outlier'].values
        print('Training size: ', len(y_train))

        if model_name == 'DeepSVDD':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        dom = torch.cat((torch.ones(n_original_train), torch.zeros(len(X_train)-n_original_train)), axis=0)
                
    
        # Test
        X_test = scaler.transform(external_test)

        if model_name == 'DAUD':
            X_train = loader(X_train, dom, DEVICE, 'train')
            
            # Test
            X_test = scaler.transform(external_test)
            X_test = loader(X_test, dom_test, DEVICE, 'test')

            model = model.to(DEVICE)

            trained_model = training_loop(model, X_train, model_name, DEVICE)

            # Compute metrics
            scores = evaluate(trained_model, model_name, X_test, DEVICE)
        else:
            X_train = get_emb_av(current_train_dataset['WSI'].values, folder)
            y_train = current_train_dataset['outlier'].values
            print('Training size: ', len(y_train))

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            
            # Test
            X_test = scaler.transform(external_test)
            model.fit(X_train)
                
            classes = np.unique(data.A_test['classes'])

            # Compute metrics
            scores = model.decision_function(X_test)

        auc_prob = calculate_metrics(labels_external, scores)
            
        # Add metrics
        metrics_bin[model_name].append(auc_prob)
        
final_metrics = {}
for model_name in model_names:
    final_metrics[model_name] = {}
    final_metrics[model_name] =  np.mean(metrics_bin[model_name]), np.std(metrics_bin[model_name]) * 0.61

print(final_metrics)


# Save final metrics
df_final = pd.DataFrame(final_metrics)
print("DataFrame Final:")
print(df_final)
save_name = "results/performance_ood.csv"
df_final.to_csv(save_name)
