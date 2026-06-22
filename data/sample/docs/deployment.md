# Deployment Guide

Services deploy to Kubernetes via Helm. CI builds an image, pushes to the registry,
and Argo CD syncs the cluster. To deploy a hotfix:

```
make build SERVICE=auth-svc
make deploy SERVICE=auth-svc ENV=staging
```

Production deploys require a green staging run plus one approval. Rollbacks use
`argocd app rollback auth-svc <revision>`. Health checks hit `/healthz` and must
return 200 within 30s or the rollout is aborted.
