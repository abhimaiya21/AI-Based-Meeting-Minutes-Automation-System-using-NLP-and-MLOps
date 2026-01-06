param (
    [Parameter(Mandatory=$true)]
    [string]$HubUsername,
    
    [string]$ImageName = "meeting-ai-backend",
    
    [string]$Tag = "latest"
)

$FullImageName = "$HubUsername/$ImageName`:$Tag"

Write-Host "[-] Building and Pushing Docker Image: $FullImageName" -ForegroundColor Cyan
# Using docker buildx build --push is more reliable for container drivers
docker buildx build --push -t $FullImageName .

if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker build/push failed. Ensure you are logged in ('docker login') and Docker Desktop is running."
    exit 1
}

Write-Host "[+] Successfully published $FullImageName" -ForegroundColor Green
