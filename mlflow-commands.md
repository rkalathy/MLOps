## Dagshub Setup

$Env:MLFLOW_TRACKING_URI = "https://dagshub.com/USERNAME/REPO.mlflow"

$Env:DAGSHUB_USERNAME    = "USERNAME"

$Env:DAGSHUB_TOKEN       = "TOKEN"

## 1. Download Miniconda (Open-Source Conda)
- **Windows 11 (64-bit):**  
  1. Go to https://docs.conda.io/en/latest/miniconda.html  
  2. Under **Windows 64-bit**, download **Miniconda3 Windows 64-bit .exe**.

- **macOS (Apple Silicon / Intel):**  
  1. Visit the same link.  
  2. Choose **Miniconda3 macOS M1 (ARM64)** or **Intel 64-bit** installer.

- **Linux (x86_64 / ARM):**  
  1. On the same page, pick the **Linux installer** that matches your CPU.

## 2. Run the Installer
- **Windows 11:**  
  ```powershell
  # Double-click the .exe and follow the prompts:
  # • Accept licence
  # • Install “Just Me”
  # • (Optional) Add to PATH
  ```
- **macOS / Linux:**  
  ```bash
  bash Miniconda3-latest-MacOSX-*.sh
  # or
  bash Miniconda3-latest-Linux-*.sh
  # Follow on-screen instructions
  ```

## 3. Initialise Conda
- **PowerShell (Win11):**  
  ```powershell
  conda init powershell
  ```
  Close & reopen PowerShell.

- **macOS / Linux (bash or zsh):**  
  ```bash
  conda init
  ```
  Restart your terminal.

## 4. Verify & Update
```bash
conda --version      # e.g. conda 23.11.0
conda update -n base -c defaults conda
```

## 5. Create & Activate an Env
```bash
conda create -n demo-ml python=3.9
conda activate demo-ml
```

## 6. Install Packages
```bash
conda install -c conda-forge numpy pandas scikit-learn mlflow boto3 sagemaker
pip install dagshub
```

---

Use `conda info --envs` to list your environments, and `conda deactivate` to exit.

## 7 Train model and send to Dagshub
python .\mlflow-demo\model.py 


Now visit Daghubs account and you will see your model has been added in MLFlow repo.




mlflow sagemaker build-and-push-container --name mlflow-pyfunc --platform aws://750952118292/us-east-2 --version 2.22.0
docker buildx build --platform linux/amd64 --tag 750952118292.dkr.ecr.eu-west-2.amazonaws.com/mlflow-pyfunc:2.22.0 --push .
docker pull mlflow/mlflow-pyfunc:2.22.0





aws configure set region eu-west-2

mlflow sagemaker build-and-push-container  --container mlflow-pyfunc --build --push



# Make sure you’re logged into ECR for eu-west-2
aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin 750952118292.dkr.ecr.eu-west-2.amazonaws.com

docker build --platform=linux/amd64 --provenance=false --output oci-mediatypes=false,type=image,push=true -t 750952118292.dkr.ecr.eu-west-2.amazonaws.com/mlflow-pyfunc:2.22.0 .

aws sagemaker-runtime invoke-endpoint --endpoint-name iris-demo-endpoint-demo-33  --body '{"instances": [[5.1, 3.5, 1.4, 0.2]]}' --content-type application/json output.json

type output.json


aws sagemaker delete-endpoint --endpoint-name iris-demo-endpoint
python mlflow-demo/deploy.py

mlflow models serve -m runs:/59d4b68a2fcd4477b7b61e532d5e6aa7/model --port 1234


https://eu-west-2.console.aws.amazon.com/sagemaker/home?region=eu-west-2#/endpoints