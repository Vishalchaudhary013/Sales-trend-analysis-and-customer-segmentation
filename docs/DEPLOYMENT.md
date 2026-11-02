# Deploying to Streamlit Community Cloud

This guide publishes the dashboard at a public `https://<name>.streamlit.app` address, deploying straight from this GitHub repository. No server administration is needed. Each time you push to the deployed branch, the app redeploys automatically.

> Streamlit's web interface changes from time to time. If a button below has moved, the official
> guide is at <https://docs.streamlit.io/deploy/streamlit-community-cloud>.

---

## 1. What the deployment needs (already in this repository)

| Item | Where | Notes |
|---|---|---|
| Entry point | `app.py` | Pages live in `pages/`; the labelled sidebar menu is rendered by `src/ui.py`. |
| Dependencies | `requirements.txt` | Pinned. Python 3.12+ uses the newest versions; 3.10/3.11 use the newest versions that support them. All are tested. |
| App settings | `.streamlit/config.toml` | Theme, 200 MB upload limit, auto-menu hidden. |
| Data | *downloaded at first start* | `data/` is not in git. On first start the app downloads the dataset (UCI, or the public CSV copy) and the exchange rates, showing progress (about 10–20 s). |
| Secrets | none | The app uses no API keys or passwords. |

---

## 2. Prepare the GitHub repository

1. Make sure the latest code is pushed. The app currently lives on the branch `claude/jolly-heisenberg-40lq4e`, which is also the repository's default branch.
2. *(Recommended)* Create a short, stable branch name for production, e.g. `main`:
   - GitHub → **Code** → branch selector → type `main` → **Create branch main from claude/jolly-heisenberg-40lq4e**.
   - Optionally make it the default: **Settings → General → Default branch**.
3. Repository visibility: Community Cloud deploys **public** repositories directly. Private repositories also work once you grant Streamlit access to them while connecting GitHub, but check Streamlit's current terms for private apps.

## 3. Create the Streamlit Community Cloud app

1. Go to <https://share.streamlit.io> and **sign in with GitHub**. Authorise the Streamlit app when asked; for a private repository, also grant access to it.
2. Click **Create app** (or **New app**) → **Deploy a public app from GitHub** (wording may vary).
3. Fill in the form:

   | Field | Value |
   |---|---|
   | Repository | `vishalchaudhary013/Palak-Project` |
   | Branch | `main` (or `claude/jolly-heisenberg-40lq4e`) |
   | Main file path | `app.py` |
   | App URL (optional) | e.g. `sales-segmentation-dashboard` |

4. Open **Advanced settings** and choose **Python 3.12**. That's the newest release line covered by the tested pins, and the version this guide assumes. Python 3.10–3.13 all work. In Community Cloud the Python version is set when the app is created; changing it later generally means deleting and redeploying the app.
5. No secrets are needed; leave the secrets box empty.
6. Click **Deploy**. The first build installs `requirements.txt`, which takes a few minutes; you can follow progress in the log panel.

## 4. First start

1. When the build finishes, the Home page opens and shows **"First start: downloading the dataset and exchange rates…"**.
2. After about 10–20 seconds it shows **536,641 transaction lines, 25,900 invoices and 4,372 customers**. That confirms the data downloaded and validated correctly.
3. Open **Early Warning** and click **"Train and save the recommended model now"**. This trains the churn model in the cloud environment, which is necessary because pickled scikit-learn models must be created with the same library version that will load them. Training takes about 30 s.
4. Work through the smoke-test checklist in the [README](../README.md#smoke-test-checklist).

## 5. Things to know about Community Cloud

- **Files are temporary.** Uploaded files, the SQLite history (`data/app.db`), saved models and exported figures are kept only until the app restarts or redeploys. After that, the dataset is downloaded again automatically and the model must be retrained (step 4.3). For permanent history, use an external database; see "Next steps" in the README.
- **Sleeping apps.** Apps that receive no traffic for a while go to sleep. The next visitor sees a wake-up screen, and the first page load repeats the data download and cache build.
- **Resources are limited.** The cleaned dataset uses about 70 MB of memory. Model cross-validation on the Model Evaluation page and "Export all figures" are the heaviest actions, at roughly 15–30 s each. Avoid uploading files close to the 200 MB limit on the free tier.
- **Network access.** The app downloads from `archive.ics.uci.edu` and, as fallbacks, `raw.githubusercontent.com`. If both are unreachable, the Home page explains how to proceed, and you can still upload a file on **Data Ingestion**.
- **Privacy.** A public app is reachable by anyone with the link. The bundled dataset is public, but don't upload real customer data to a public deployment. Use **App settings → Sharing** to restrict viewers if your plan allows it, or deploy privately.

## 6. Updating the app

```bash
git add -A
```

```bash
git commit -m "Describe the change"
```

```bash
git push origin main
```

Community Cloud picks up the push and redeploys within a minute or two. If you changed `requirements.txt`, the redeploy reinstalls dependencies.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails with "No matching distribution found for pandas==3.1.0" | Python older than 3.12 was selected, but `requirements.txt` handles that with version markers. Check you deployed the latest `requirements.txt`. If it still fails, redeploy with Python 3.12. |
| Home page: "The bundled dataset is not available" | The first-start download failed (network). Reboot the app (**⋮ → Reboot app**); the download is retried. Alternatively upload the CSV on **Data Ingestion**. |
| Early Warning: "trained with scikit-learn X … Retrain it here" | Expected after a dependency change. Click **Train and save the recommended model now**. |
| App is slow on first visit | First-time download, cleaning and cross-validation; later visits use the caches. |
| Logs | **Manage app** (bottom right of the running app) → logs. Errors in the UI are shown as friendly messages; full details are written to these logs. |

## 8. Running locally instead

See **Setup** and **Usage** in the [README](../README.md#setup): create a virtual environment, `pip install -r requirements.txt`, then `streamlit run app.py`. The same first-start download happens automatically, or run `python scripts/fetch_data.py` beforehand.
