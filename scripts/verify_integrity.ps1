param([string]$Manifest = (Join-Path $PSScriptRoot '../SHA256SUMS.txt'))
$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path (Split-Path -Parent $PSCommandPath) -Parent
$taskVerifiedCount = 0
foreach ($taskManifestLine in Get-Content -LiteralPath $Manifest) {
    if (-not $taskManifestLine.Trim()) { continue }
    if ($taskManifestLine -notmatch '^([0-9a-f]{64})  (.+)$') { throw 'Invalid checksum manifest line' }
    $taskExpectedHash = $matches[1]
    $taskRelativePath = $matches[2]
    $taskFilePath = Join-Path $taskProjectRoot $taskRelativePath
    if (-not (Test-Path -LiteralPath $taskFilePath -PathType Leaf)) { throw "Missing file: $taskRelativePath" }
    $taskActualHash = (Get-FileHash -LiteralPath $taskFilePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskActualHash -ne $taskExpectedHash) { throw "Checksum mismatch: $taskRelativePath (run git lfs pull for checkpoint files)" }
    $taskVerifiedCount++
}
Write-Output "Verified $taskVerifiedCount files (SHA256)."
