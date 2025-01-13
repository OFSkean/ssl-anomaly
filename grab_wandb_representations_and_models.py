import wandb
import os
import pandas as pd


project_names = [
    'ssl-4-anomaly/medianomaly_mae_ablation',
    'ssl-4-anomaly/medianomaly_dino_ablation',
    'ssl-4-anomaly/medianomaly_bt_ablation',
    'ssl-4-anomaly/medianomaly_simclr_ablation',
    'ssl-4-anomaly/medianomaly_csi_ablation',
    
    'ssl-4-anomaly/visa_simclr_ablation',
    'ssl-4-anomaly/visa_ad_dino_ablation',
    'ssl-4-anomaly/visa_bt_ablation',
    'ssl-4-anomaly/visa_mae_ablation',
    'ssl-4-anomaly/visa_csi_ablation',

    'ssl-4-anomaly/sewerml_csi_ablation',
    'ssl-4-anomaly/sewerml_dino_ablation',
    'ssl-4-anomaly/sewerml_bt_ablation',
    'ssl-4-anomaly/sewerml_simclr_ablation',
    'ssl-4-anomaly/sewerml_mae_ablation',

    'ssl-4-anomaly/mvtec_ad_dino_ablation',
    'ssl-4-anomaly/mvtec_ad_csi_ablation',
    'ssl-4-anomaly/mvtec_ad_bt_ablation',
    'ssl-4-anomaly/mvtec_ad_simclr_ablation',
    'ssl-4-anomaly/mvtec_ad_mae_ablation',
]

def get_dataset_name(project_name):
    return project_name.split('/')[-1].split('_')[0]

def get_model_name(project_name):
    return project_name.split('/')[-1].split('_')[1]

for project_name in project_names:
    try:
        print(project_name)
        dataset_name = get_dataset_name(project_name)
        model_name = get_model_name(project_name)

        local_dir = f"data/representations/{dataset_name}/{model_name}"
        os.makedirs(local_dir, exist_ok=True)

        runs = wandb.Api().runs(project_name)
        for run in runs:
            print('.',run.name)
            run_dir = f"{local_dir}/{run.name}"
                
            # download the features folder
            os.makedirs(run_dir, exist_ok=True)
            for file in run.files():
                # skip the file if it already exists
                if os.path.exists(f"{run_dir}/{file.name}"):
                    continue

                if 'features' in file.name:
                    print('...', file.name)
                    file.download(root=run_dir)
                if 'last.pth' in file.name:
                    print('...', file.name)
                    file.download(root=run_dir)

            # save the accuracy metrics to a csv file
            metrics = dict(run.summary)
            accuracy_metrics = {k: v for k, v in metrics.items() if 'accuracy' in k or 'f1' in k or 'precision' in k or 'recall' in k}
            accuracy_metrics_df = pd.DataFrame(accuracy_metrics, index=[0])
            accuracy_metrics_df.to_csv(f"{run_dir}/accuracies.csv")
    except Exception as e:
        print(e)
        continue
