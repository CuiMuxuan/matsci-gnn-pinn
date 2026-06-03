param(
    [string]$HostName = "223.109.239.30",
    [int]$Port = 22036,
    [string]$User = "root",
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\matsci_gnnpinn_a100",
    [string]$RemoteRepo = "/root/matsci-gnn-pinn",
    [string]$LocalOutputDir = "docs/results/phase103_nist_ammt_registered_intake",
    [string]$LocalLogDir = "logs",
    [int64]$MaxArtifactBytes = 52428800
)

$ErrorActionPreference = "Stop"

function Invoke-ScpPull {
    param(
        [string]$RemotePath,
        [string]$LocalPath
    )
    $localParent = Split-Path -Parent $LocalPath
    if ($localParent) {
        New-Item -ItemType Directory -Force -Path $localParent | Out-Null
    }
    $remoteSpec = "${User}@${HostName}:$RemoteRepo/$RemotePath"
    $remoteFile = "$RemoteRepo/$RemotePath"
    $sizeText = ssh -i $IdentityFile -p $Port -o IdentitiesOnly=yes "${User}@${HostName}" "stat -c %s '$remoteFile'"
    $remoteBytes = [int64]($sizeText | Select-Object -First 1)
    if ($remoteBytes -gt $MaxArtifactBytes) {
        throw "Refusing to pull oversized Phase 103 artifact ($remoteBytes bytes > $MaxArtifactBytes): $remotePath"
    }
    scp -i $IdentityFile -P $Port -o IdentitiesOnly=yes $remoteSpec $LocalPath
}

$artifactFiles = @(
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_file_audit.csv",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_zip_member_keywords.csv",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_registered_intake_gate.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_registered_intake_manifest.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_schema_scout_summary.csv",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_schema_scout_candidates.csv",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_schema_scout_gate.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_schema_scout_manifest.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_member_schema_samples.csv",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_member_schema_sampler_gate.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_member_schema_sampler_manifest.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_deep_sequence_groups.csv",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_deep_target_binary_samples.csv",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_deep_timing_evidence.csv",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_deep_registration_probe_gate.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_deep_registration_probe_manifest.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_source_target_join_candidates.csv",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_join_probe_gate.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_join_probe_manifest.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_tiny_table_feasibility_roles.csv",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_tiny_table_feasibility_gate.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_tiny_table_feasibility_summary.md",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_tiny_table_feasibility_manifest.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_tiny_registered_source_target_table.csv",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_tiny_registered_split_manifest.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_tiny_registered_table_gate.json",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_tiny_registered_table_summary.md",
    "docs/results/phase103_nist_ammt_registered_intake/phase103_nist_ammt_tiny_registered_table_manifest.json"
)

$logFiles = @(
    "logs/phase103_nist_ammt_intake_a100_manifest.json",
    "logs/phase103_nist_ammt_schema_scout_a100_manifest.json",
    "logs/phase103_nist_ammt_member_schema_sampler_a100_manifest.json",
    "logs/phase103_nist_ammt_deep_registration_probe_a100_manifest.json",
    "logs/phase103_nist_ammt_join_probe_a100_manifest.json",
    "logs/phase103_nist_ammt_tiny_table_feasibility_a100_manifest.json",
    "logs/phase103_nist_ammt_tiny_registered_table_a100_manifest.json",
    "logs/phase103_nist_ammt_triage_watch.log"
)

foreach ($remote in $artifactFiles) {
    $name = Split-Path -Leaf $remote
    Invoke-ScpPull -RemotePath $remote -LocalPath (Join-Path $LocalOutputDir $name)
}

foreach ($remote in $logFiles) {
    $name = Split-Path -Leaf $remote
    Invoke-ScpPull -RemotePath $remote -LocalPath (Join-Path $LocalLogDir $name)
}

$gatePaths = @(
    (Join-Path $LocalOutputDir "phase103_nist_ammt_registered_intake_gate.json"),
    (Join-Path $LocalOutputDir "phase103_nist_ammt_schema_scout_gate.json"),
    (Join-Path $LocalOutputDir "phase103_nist_ammt_member_schema_sampler_gate.json"),
    (Join-Path $LocalOutputDir "phase103_nist_ammt_deep_registration_probe_gate.json"),
    (Join-Path $LocalOutputDir "phase103_nist_ammt_join_probe_gate.json"),
    (Join-Path $LocalOutputDir "phase103_nist_ammt_tiny_table_feasibility_gate.json"),
    (Join-Path $LocalOutputDir "phase103_nist_ammt_tiny_registered_table_gate.json")
)

foreach ($path in $gatePaths) {
    if (Test-Path $path) {
        $gate = Get-Content -Raw $path | ConvertFrom-Json
        [PSCustomObject]@{
            gate = Split-Path -Leaf $path
            status = $gate.status
            phase104_baseline_smoke_allowed = $gate.phase104_baseline_smoke_allowed
            a100_training_allowed_now = $gate.a100_training_allowed_now
        }
    }
}
