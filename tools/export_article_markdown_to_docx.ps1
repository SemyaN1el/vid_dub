param(
    [Parameter(Mandatory = $true)]
    [string]$MarkdownPath,

    [Parameter(Mandatory = $true)]
    [string]$DocxPath
)

$ErrorActionPreference = "Stop"

function Clean-MarkdownText {
    param([string]$Text)
    $value = $Text -replace "`r", ""
    $value = $value -replace "\*\*", ""
    $value = $value.Replace([string][char]96, "")
    if ($value.StartsWith("*") -and $value.EndsWith("*") -and $value.Length -gt 1) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    return $value.Trim()
}

$mdFull = [System.IO.Path]::GetFullPath($MarkdownPath)
$docxFull = [System.IO.Path]::GetFullPath($DocxPath)
$baseDir = [System.IO.Path]::GetDirectoryName($mdFull)
$docxDir = [System.IO.Path]::GetDirectoryName($docxFull)

if (-not (Test-Path $mdFull)) {
    throw "Markdown file not found: $mdFull"
}

if (-not (Test-Path $docxDir)) {
    New-Item -ItemType Directory -Path $docxDir | Out-Null
}

$word = $null
$doc = $null
$selection = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $doc = $word.Documents.Add()
    $selection = $word.Selection

    $doc.PageSetup.TopMargin = $word.CentimetersToPoints(2)
    $doc.PageSetup.BottomMargin = $word.CentimetersToPoints(2)
    $doc.PageSetup.LeftMargin = $word.CentimetersToPoints(2)
    $doc.PageSetup.RightMargin = $word.CentimetersToPoints(2)

    $normalIndent = $word.CentimetersToPoints(1.25)
    $lines = Get-Content $mdFull -Encoding UTF8

    foreach ($rawLine in $lines) {
        $line = $rawLine.TrimEnd()

        if ([string]::IsNullOrWhiteSpace($line)) {
            $selection.TypeParagraph()
            continue
        }

        $selection.Font.Name = "Times New Roman"
        $selection.Font.Bold = 0
        $selection.Font.Italic = 0
        $selection.Font.Size = 14
        $selection.ParagraphFormat.FirstLineIndent = $normalIndent
        $selection.ParagraphFormat.Alignment = 3
        $selection.ParagraphFormat.SpaceAfter = 6
        $selection.ParagraphFormat.LineSpacingRule = 0

        if ($line.StartsWith("![")) {
            if ($line -match "!\[[^\]]*\]\(([^)]+)\)") {
                $imageRel = $matches[1]
                $imagePath = [System.IO.Path]::GetFullPath((Join-Path $baseDir $imageRel))
                $selection.ParagraphFormat.Alignment = 1
                $selection.ParagraphFormat.FirstLineIndent = 0
                $shape = $selection.InlineShapes.AddPicture($imagePath, $false, $true)
                $shape.Width = $word.CentimetersToPoints(15)
                $selection.TypeParagraph()
            }
            continue
        }

        if ($line.StartsWith("# ")) {
            $selection.ParagraphFormat.Alignment = 1
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.Font.Bold = 1
            $selection.Font.Size = 14
            $selection.TypeText((Clean-MarkdownText $line.Substring(2)))
            $selection.TypeParagraph()
            continue
        }

        if ($line.StartsWith("УДК ")) {
            $selection.ParagraphFormat.Alignment = 0
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.TypeText($line)
            $selection.TypeParagraph()
            continue
        }

        if ($line.StartsWith("**[Ф.") -or $line.StartsWith("**[Ф")) {
            $selection.ParagraphFormat.Alignment = 1
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.Font.Bold = 1
            $selection.TypeText((Clean-MarkdownText $line))
            $selection.TypeParagraph()
            continue
        }

        if ($line -eq "[Название образовательной организации]") {
            $selection.ParagraphFormat.Alignment = 1
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.TypeText($line)
            $selection.TypeParagraph()
            continue
        }

        if ($line.StartsWith("**Аннотация.**")) {
            $selection.Font.Size = 12
            $selection.ParagraphFormat.Alignment = 3
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.Font.Bold = 1
            $selection.TypeText("Аннотация.")
            $selection.Font.Bold = 0
            $selection.TypeText(" " + (Clean-MarkdownText $line.Replace("**Аннотация.**", "")))
            $selection.TypeParagraph()
            continue
        }

        if ($line.StartsWith("**Ключевые слова:**")) {
            $selection.Font.Size = 12
            $selection.ParagraphFormat.Alignment = 3
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.Font.Bold = 1
            $selection.TypeText("Ключевые слова:")
            $selection.Font.Bold = 0
            $selection.TypeText(" " + (Clean-MarkdownText $line.Replace("**Ключевые слова:**", "")))
            $selection.TypeParagraph()
            continue
        }

        if ($line -eq "Список литературы") {
            $selection.ParagraphFormat.Alignment = 0
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.Font.Bold = 0
            $selection.TypeText($line)
            $selection.TypeParagraph()
            continue
        }

        if ($line.StartsWith("*Рис.")) {
            $selection.ParagraphFormat.Alignment = 1
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.Font.Size = 12
            $selection.Font.Italic = 1
            $selection.TypeText((Clean-MarkdownText $line))
            $selection.TypeParagraph()
            continue
        }

        if ($line -match "^\[\d+\]") {
            $selection.ParagraphFormat.Alignment = 3
            $selection.ParagraphFormat.FirstLineIndent = 0
            $selection.Font.Size = 14
            $selection.TypeText((Clean-MarkdownText $line))
            $selection.TypeParagraph()
            continue
        }

        $selection.TypeText((Clean-MarkdownText $line))
        $selection.TypeParagraph()
    }

    $doc.SaveAs([ref]$docxFull, [ref]16)
    $doc.Close()
    $word.Quit()
}
finally {
    if ($doc -ne $null) {
        try { $doc.Close() } catch {}
    }
    if ($word -ne $null) {
        try { $word.Quit() } catch {}
    }
}

Write-Output $docxFull
