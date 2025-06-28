KUBECTL_VERSION=$(curl -sL https://dl.k8s.io/release/stable.txt)

echo "$KUBECTL_VERSION"

curl -Lo kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/
kubectl version --client

curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64

chmod +x minikube

sudo mv minikube /usr/local/bin/

minikube version

sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker $USER
newgrp docker 

minikube start --driver=docker

kubectl get nodes

kubectl create namespace kubeflow

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=1.8.5"


kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io


kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=1.8.5"

kubectl get pods -n kubeflow --watch
