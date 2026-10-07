# Runs the remaining full-trainval steps once the radar points cache has finished.
# Stops at the first failing step. Log: D:\radar-bev-llm\data\pipeline_log.txt

$py = "D:\miniconda3\envs\rbl\python.exe"
$log = "D:\radar-bev-llm\data\pipeline_log.txt"
Set-Location "D:\radar-bev-llm\code\radar"

function Log($msg) { "$(Get-Date -Format 'HH:mm:ss') $msg" | Add-Content $log }

function Run-Step($name, $args_) {
    Log "START $name"
    & $py @args_
    if ($LASTEXITCODE -ne 0) { Log "FAILED $name (exit $LASTEXITCODE)"; exit 1 }
    Log "DONE $name"
}

Log "waiting for radar points cache to finish"
while (-not (Select-String -Path "D:\radar-bev-llm\data\points_cache_log.txt" -Pattern "POINTSDONE" -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
Log "points cache finished"

Run-Step "motion-token cache" @("make_motion_cache.py", "--version", "v1.0-trainval", "--tag", "trainval_obj")

Run-Step "radar detector training (full train split)" @("train_radar.py", "--version", "v1.0-trainval", "--train-split", "train", "--val-split", "val", "--mirror", "--epochs", "8", "--gpu", "0", "--out", "D:/radar-bev-llm/checkpoints/radar_pillar_trainval.pt")

Run-Step "BEV feature cache (all keyframes)" @("make_bev_cache.py", "--version", "v1.0-trainval", "--splits", "train", "val", "--tag", "trainval_bev", "--ckpt", "D:/radar-bev-llm/checkpoints/radar_pillar_trainval.pt", "--gpu", "0")

Run-Step "language training (fusion + object tokens, subsampled)" @("train_qa.py", "--name", "fusion_obj_trainval", "--feat-tag", "trainval_bev", "--camera-tag", "trainval_clip", "--motion-tag", "trainval_obj", "--subsample", "60000", "--val-subsample", "6000", "--epochs", "2", "--bs", "8", "--gpu", "0")

Log "PIPELINE ALL DONE"
