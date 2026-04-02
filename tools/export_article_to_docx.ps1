param(
    [Parameter(Mandatory = $true)]
    [string]$HtmlPath,

    [Parameter(Mandatory = $true)]
    [string]$DocxPath
)

$ErrorActionPreference = "Stop"

$htmlFull = [System.IO.Path]::GetFullPath($HtmlPath)
$docxFull = [System.IO.Path]::GetFullPath($DocxPath)
$docxDir = [System.IO.Path]::GetDirectoryName($docxFull)

if (-not (Test-Path $htmlFull)) {
    throw "HTML file not found: $htmlFull"
}

if (-not (Test-Path $docxDir)) {
    New-Item -ItemType Directory -Path $docxDir | Out-Null
}

$word = $null
$document = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $document = $word.Documents.Open($htmlFull)
    $document.SaveAs([ref]$docxFull, [ref]16)
    $document.Close()
    $word.Quit()
}
finally {
    if ($document -ne $null) {
        try { $document.Close() } catch {}
    }
    if ($word -ne $null) {
        try { $word.Quit() } catch {}
    }
}

Write-Output $docxFull
