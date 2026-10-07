# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [SemVer](https://semver.org/).

## [2.0.0] - 2026-10-07

First tagged release. Latest changes:

- docs: add colors to mermaid diagrams (#13)
- docs: typography pass, no em dash or middle dot (#12)
- feat(api): add Render free-tier keepalive self-ping (#11)
- fix: skip bootstrap import chain when MPI_SKIP_BOOTSTRAP is set (#10)
- fix: resolve frozen-mode fork bomb and add Render deployment (#9)
- refactor: reorganize src/ by business domain (data/models/validation/analysis) (#8)
- docs: add JURY_GUIDE.md for jury navigation by business need (#7)
- fix(ports): use env vars MPI_API_PORT=8001 MPI_DASH_PORT=8502
- feat(imbalance): add standalone script 16 + headless imbalance module with 5 strategies (#6)
- feat(imbalance): add SMOTE, threshold optimization, stratified CV notebook (#5)
- build: regenerate PDF report + PPTX with real metrics (XGBoost F1=0.886 ROC-AUC=0.995) (#4)
- fix(readme): replace example metrics with real values from reports/03/metrics_summary.json (#3)
- feat: bonus maintenance · drift PSI, conformal prediction, MLflow, robustesse bruit (#2)
- style: passage du dashboard à la DA EFREI (#1)
- docs: add star history chart to readme
