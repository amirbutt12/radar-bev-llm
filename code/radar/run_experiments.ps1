# Trainval baselines on the same subsamples as fusion_obj_trainval, then condition-split evaluation for all three.
# Stops at the first failure. Log: D:\radar-bev-llm\data\experiments_log.txt

$py = "D:\miniconda3\envs\rbl\python.exe"
$log = "D:\radar-bev-llm\data\experiments_log.txt"
Set-Location "D:\radar-bev-llm\code\radar"
$common = @("--feat-tag", "trainval_bev", "--subsample", "60000", "--val-subsample", "6000", "--epochs", "2", "--bs", "8", "--gpu", "0")

function Log($msg) { "$(Get-Date -Format 'HH:mm:ss') $msg" | Add-Content $log }

function Run-Step($name, $args_) {
    Log "START $name"
    $steplog = "D:\radar-bev-llm\data\step_" + ($name -replace '[^a-zA-Z0-9]+', '_') + ".log"
    & $py @args_ *> $steplog
    if ($LASTEXITCODE -ne 0) { Log "FAILED $name (exit $LASTEXITCODE), see $steplog"; exit 1 }
    Log "DONE $name"
}

Run-Step "radar-only trainval" (@("train_qa.py", "--name", "radar_trainval") + $common)
Run-Step "blind trainval" (@("train_qa.py", "--name", "blind_trainval", "--no-vision") + $common)
Run-Step "conditions fusion_obj_trainval" @("evaluate_conditions.py", "--name", "fusion_obj_trainval", "--gpu", "0")
Run-Step "conditions radar_trainval" @("evaluate_conditions.py", "--name", "radar_trainval", "--gpu", "0")
Run-Step "conditions blind_trainval" @("evaluate_conditions.py", "--name", "blind_trainval", "--gpu", "0")

Log "EXPERIMENTS ALL DONE"
