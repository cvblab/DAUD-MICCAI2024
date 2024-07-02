# Other methods from PYOD

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
n_runs = 10

model_names = ['DAUD', 'ECOD', 'DeepSVDD', 'AnoGAN', 'VAE', 'beta-VAE']
# Dict to save the metrics. For each institution, we fill two list (i.e., avg and std results).
metrics = {}
for model_name in model_names:
    metrics[model_name] = {'A':np.zeros((6,n_runs)), 'B': np.zeros((6,n_runs))}

metrics_bin = {}
for model_name in model_names:
    metrics_bin[model_name] = {'A':np.zeros((n_runs)), 'B': np.zeros((n_runs))}

# Data
data = data_partition(CSV_file, 0)

# Test set
X_test_A = get_emb_av(data.A_test['values'], folder)
X_test_B = get_emb_av(data.B_test['values'], folder)

# Train set
train_dataset = pd.DataFrame({'WSI': data.train_config['values'], 'outlier':  data.train_config['labels']})    
n_original_train = len(train_dataset)

for run_ix in range(n_runs):    
    num_samples_to_add = len(data.train_config['perc_values'])
    subset = data.train_config['perc_values'][:num_samples_to_add]
    labels_subset = data.train_config['perc_labels'][:num_samples_to_add]
    current_train_dataset = pd.concat([train_dataset, pd.DataFrame({'WSI': subset, 'outlier': labels_subset})])
   

    for model_name in model_names:
        print(model_name)
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
        X_test_A = get_emb_av(data.A_test['values'], folder)
        X_test_B = get_emb_av(data.B_test['values'], folder)
        X_test_A, X_test_B = scaler.transform(X_test_A),scaler.transform(X_test_B)
        # Model
        model = select_model(model_name)

        if model_name == 'DAUD':
            X_train = loader(X_train, dom, DEVICE, 'train')
            
            # Test
            X_test_A = get_emb_av(data.A_test['values'], folder)
            X_test_B = get_emb_av(data.B_test['values'], folder)
            X_test_A, X_test_B = scaler.transform(X_test_A),scaler.transform(X_test_B)
            X_test_A, X_test_B = loader(X_test_A, data.dom_test_A, DEVICE, 'test'), loader(X_test_B, data.dom_test_B, DEVICE, 'test')  

            model = model.to(DEVICE)

            trained_model = training_loop(model, X_train, model_name, DEVICE)

            # Compute metrics
            scores_A = evaluate(trained_model, model_name, X_test_A, DEVICE)
            scores_B = evaluate(trained_model, model_name, X_test_B, DEVICE)

        else:
            model.fit(X_train)
            
            classes = np.unique(data.A_test['classes'])

            # Compute metrics
            scores_A = model.decision_function(X_test_A)
            scores_B = model.decision_function(X_test_B)

        auc_A = calculate_metrics(data.A_test['labels'], scores_A)
        auc_B = calculate_metrics(data.B_test['labels'], scores_B)

        # Add metrics
        metrics_bin[model_name]['A'][run_ix] = auc_A
        metrics_bin[model_name]['B'][run_ix] = auc_B


    final_metrics_bin = {}
    for model_name in model_names:
        final_metrics_bin[model_name] = {}
        final_metrics_bin[model_name]['A'] =  metrics_bin[model_name]['A'].mean(), metrics_bin[model_name]['A'].std() * 0.61
        final_metrics_bin[model_name]['B'] =  metrics_bin[model_name]['B'].mean(), metrics_bin[model_name]['B'].std() * 0.61

 
    dataframes_resultantes = {}


    for clave, subdiccionario in final_metrics_bin.items():
        df = pd.DataFrame({f'{clave}_{subclave}': vector for subclave, vector in subdiccionario.items()})
        dataframes_resultantes[clave] = df

    df_final = pd.concat(dataframes_resultantes.values(), axis=1)

    # Save the final results
    print("DataFrame Final:")
    print(df_final)

    save_name = "results/performance_comparison.csv"
    df_final.to_csv(save_name)
