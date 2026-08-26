param(
    [string]$Path = ".",
    [string[]]$Exclude = @("venv")
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

function Show-Tree {
    param(
        [string]$CurrentPath,
        [string]$Indent = ""
    )

    $items = Get-ChildItem -Path $CurrentPath -Force | Where-Object {
        $item = $_
        $excluded = $false
        foreach ($ex in $Exclude) {
            if ($item.Name -eq $ex) {
                $excluded = $true
                break
            }
        }
        -not $excluded
    }

    $count = $items.Count
    for ($i = 0; $i -lt $count; $i++) {
        $item = $items[$i]
        $isLast = ($i -eq $count - 1)
        
        $connector = if ($isLast) { "`-- " } else { "|-- " }
        
        Write-Output "$Indent$connector$($item.Name)"
        
        if ($item.PSIsContainer) {
            $newIndent = if ($isLast) { "$Indent    " } else { "$Indent|   " }
            Show-Tree -CurrentPath $item.FullName -Indent $newIndent
        }
    }
}

Write-Output $Path
Show-Tree -CurrentPath $Path