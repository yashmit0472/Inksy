# Deploy Inksy on Render

The public demo serves the standard handwriting style and PNG export. Personal style training is disabled unless you set an admin password. The base model is loaded from a private Render secret file and checked against its SHA-256 before the app starts. The dataset and checkpoint stay out of Git.

1. Push the deployment branch to GitHub. In the [Render dashboard](https://dashboard.render.com/), create a Blueprint from that branch using `render.yaml`. The Blueprint uses a free Python web service, Python 3.12, and `/api/status` for health checks.
2. On the new service's **Environment** page, add the base64-encoded standard checkpoint as a secret file named `inksy-model.b64`. Use **Save and deploy**. The first attempt can fail while the secret file is missing. The service picks it up when the next deploy starts. The pinned model is the local `artifacts/print-handwriting-v3-balanced/model.pt`.
3. Open the deployed URL. Generate `Meet me in Room B-27 at 9:30.` and download the PNG. Also confirm that `/api/status` returns `ready: true` and the public visitor cannot load `/train`.

To prepare the model text on Windows from the repo root:

```powershell
$model = (Resolve-Path 'artifacts/print-handwriting-v3-balanced/model.pt').Path
$encoded = [Convert]::ToBase64String([IO.File]::ReadAllBytes($model))
[IO.File]::WriteAllText((Join-Path (Get-Location) 'artifacts/inksy-model.b64'), $encoded)
```

For owner access, set a strong `INKSY_ADMIN_PASSWORD` in Render and open `/train` over HTTPS using the username `inksy`. Uploads, training status, and saved personal styles are available to signed-in users. To keep personal styles across redeploys, attach a persistent disk and set `INKSY_STYLE_DIR` to a directory on it. Persistent disks require a paid service. Use one instance so the current training job and login session stay on the same process.

The standard model uses IAM and EMNIST samples. The [IAM release](https://zenodo.org/records/19003721) describes noncommercial research use and requests attribution. Keep the portfolio deployment noncommercial and confirm the dataset terms before changing the use of the demo.

## Public demo on Vercel

Vercel serves the same public studio and PNG export. The package contains the standard model's precomputed glyph images and a Python function that lays them out with the same renderer as Render. The function does not need PyTorch. Keep the generated package under ignored `artifacts/`. It does not contain model weights, datasets, or personal styles.

```powershell
.\.venv\Scripts\python.exe -m deploy.prepare_vercel artifacts/vercel-public
```

Deploy the contents of `artifacts/vercel-public` as the project root. The Vercel configuration serves only `public/` as static files and bundles the glyph images into the render function. Verify `/api/status` and render a sentence with mixed case and digits. Train and save personal styles through the local studio.
