
$targetString = "emnist_c10" # 替换为您要查找的字符串

# 查找含有目标字符串的进程
$processes = Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*$targetString*" }

# 关闭这些进程
$processes | ForEach-Object {
    Write-Host "Closing process $($_.Name) with PID $($_.ProcessId)..."
    Stop-Process -Id $_.ProcessId -Force
}

Write-Host "Done."
