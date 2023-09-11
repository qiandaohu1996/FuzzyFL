import os
import subprocess


def execute_command_in_files(folder_path, command):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                print(file)
                file_path = os.path.join(root, file)
                subprocess.run(command + " " + file_path, shell=True)


# 设置文件夹路径和命令
current_dir = os.getcwd()
utils_dir = os.path.join(current_dir, "utils")
command = "python -m yapf --style .style.yapf --in-place"

# 执行命令
execute_command_in_files(utils_dir, command)
