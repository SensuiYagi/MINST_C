import subprocess

# ユーザーからの入力を受け取る
com_port = "COM3"
baud_rate = "9600"

# Cプログラムの実行
subprocess.run(["C:/Users/sensu/source/repos/serial_reader/out/build/x64-debug/serial_reader.exe", com_port, baud_rate])

