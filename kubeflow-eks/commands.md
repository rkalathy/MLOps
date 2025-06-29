windows
New-Item -ItemType Directory -Path "$env:USERPROFILE\.ssh"
ssh-keygen -t rsa -b 2048 -f "$env:USERPROFILE\.ssh\id_rsa"

Mac/linux
ssh-keygen -t rsa -b 2048 -f "~\.ssh\id_rsa"

Login ECR
aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin 750952118292.dkr.ecr.eu-west-2.amazonaws.com

docker tag kubeflow-train:latest 750952118292.dkr.ecr.eu-west-2.amazonaws.com/kubeflow-train:latest

docker push 750952118292.dkr.ecr.eu-west-2.amazonaws.com/kubeflow-train:latest
