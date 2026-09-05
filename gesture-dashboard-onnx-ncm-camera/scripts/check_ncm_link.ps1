param(
    [string]$HostIp = "192.168.50.1",
    [string]$DeviceIp = "192.168.50.2",
    [int]$TcpPort = 5000
)

$ErrorActionPreference = "Stop"

Write-Host "USB-NCM host configuration" -ForegroundColor Cyan
$HostAddresses = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -eq $HostIp }
if ($null -eq $HostAddresses -or @($HostAddresses).Count -eq 0) {
    Write-Host "FAIL: Windows does not currently own $HostIp." -ForegroundColor Red
    Write-Host "Configure the NCM adapter as $HostIp with prefix length 30."
} else {
    $HostAddresses | Format-Table InterfaceAlias, InterfaceIndex, IPAddress, PrefixLength, AddressState -AutoSize
    if (@($HostAddresses | Where-Object PrefixLength -eq 30).Count -eq 0) {
        Write-Host "WARNING: $HostIp exists, but it is not configured with /30." -ForegroundColor Yellow
    } else {
        Write-Host "PASS: host address $HostIp/30 exists." -ForegroundColor Green
    }
}

Write-Host "ARP/neighbor state for the development board" -ForegroundColor Cyan
$Neighbor = Get-NetNeighbor -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -eq $DeviceIp }
if ($null -eq $Neighbor -or @($Neighbor).Count -eq 0) {
    Write-Host "No neighbor-table entry exists for $DeviceIp yet." -ForegroundColor Yellow
} else {
    $Neighbor | Format-Table InterfaceAlias, IPAddress, LinkLayerAddress, State -AutoSize
}

Write-Host "TCP JLIP service test" -ForegroundColor Cyan
$TcpResult = Test-NetConnection -ComputerName $DeviceIp -Port $TcpPort -InformationLevel Detailed
$TcpResult | Select-Object ComputerName, RemoteAddress, RemotePort, SourceAddress, InterfaceAlias, TcpTestSucceeded |
    Format-List
if ($TcpResult.TcpTestSucceeded) {
    Write-Host "PASS: the board accepts TCP connections on $DeviceIp`:$TcpPort." -ForegroundColor Green
} else {
    Write-Host "FAIL: TCP SYN did not complete. Check ARP and firmware tcp accept($TcpPort)." -ForegroundColor Red
}

Write-Host "Run UDP discovery from the dashboard or POST http://127.0.0.1:8200/api/ncm/discover." -ForegroundColor Cyan
