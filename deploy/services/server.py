import subprocess

subprocess.run(["grafana", "server"])
subprocess.run(["alertmanager"])
subprocess.run(["node_exporter"])
subprocess.run(["prometheus", "--config.file=prometheus.yml"])
subprocess.run(["python3", "src/sentiments/server.py"])
