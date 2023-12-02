import serial.tools.list_ports

# 利用可能なCOMポートのリストを取得
ports = serial.tools.list_ports.comports()

# 利用可能なポートがある場合、それらを表示
if ports:
    for port in ports:
        print(f"Port: {port.device}, Description: {port.description}")
else:
    print("No COM ports found.")
