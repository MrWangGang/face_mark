import subprocess
import sys

if __name__ == "__main__":
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "server.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动 Streamlit 时出错: {e}")
