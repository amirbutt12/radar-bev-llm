# Finishes the motion-question ablation matrix and day/night evaluation, one GPU job at a time.
# Resilient: a failing step is logged and skipped, not fatal, so the rest of the night isn't wasted.
# Log: D:\radar-bev-llm\data\ablations_log.txt

$py = "D:\miniconda3\envs\rbl\python.exe"
$log = "D:\radar-bev-llm\data\ablations_log.txt"
Set-Location "D:\radar-bev-llm\code\radar"
$common = @("--feat-tag", "trainval_bev", "--motion-qa", "--motion-qa-tag", "motion_trainval", "--subsample", "60000", "--val-subsample", "6000", "--epochs", "2", "--bs", "8", "--gpu", "0")

function Log($msg) { "$(Get-Date -Format 'HH:mm:ss') $msg" | Add-Content $log }

function Run-Step($name, $args_) {
    Log "START $name"
    $steplog = "D:\radar-bev-llm\data\step_" + ($name -replace '[^a-zA-Z0-9]+', '_') + ".log"
    & $py @args_ *> $steplog
    if ($LASTEXITCODE -ne 0) { Log "FAILED $name (exit $LASTEXITCODE), see $steplog -- continuing to next step"; return $false }
    Log "DONE $name"
    return $true
}

Log "waiting for motion_only_trainval_motionqa to finish"
while (-not (Select-String -Path "D:\radar-bev-llm\data\motion_only_motionqa.log" -Pattern "EXIT" -Quiet -ErrorAction SilentlyContinue)) {
    Start-Sleep -Seconds 60
}
Log "motion_only_trainval_motionqa finished (was started before this script)"

# fills out the modality ladder on the motion questions: dense radar only, camera only, no sensors
Run-Step "radar trainval motionqa" (@("train_qa.py", "--name", "radar_trainval_motionqa") + $common) | Out-Null
Run-Step "camera trainval motionqa" (@("train_qa.py", "--name", "camera_trainval_motionqa", "--camera-tag", "trainval_clip") + $common) | Out-Null
Run-Step "blind trainval motionqa" (@("train_qa.py", "--name", "blind_trainval_motionqa", "--no-vision") + $common) | Out-Null

# day/night/rain breakdown for every motion-qa model that has a checkpoint
foreach ($name in "fusion_obj_trainval_motionqa", "radar_obj_trainval_motionqa", "motion_only_trainval_motionqa",
                   "radar_trainval_motionqa", "camera_trainval_motionqa", "blind_trainval_motionqa") {
    if (Test-Path "D:\radar-bev-llm\checkpoints\qa_$name.pt") {
        Run-Step "conditions $name" @("evaluate_conditions.py", "--name", $name, "--motion-qa-tag", "motion_trainval", "--gpu", "0") | Out-Null
    } else {
        Log "SKIP conditions $name -- no checkpoint"
    }
}

Log "ABLATIONS ALL DONE"
