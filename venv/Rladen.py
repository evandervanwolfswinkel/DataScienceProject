import os
import subprocess
from subprocess import Popen

heatmap = os.system('/interactive heatmap.R')
p = subprocess.Popen(['Rscript', '/interactive heatmap.R'], stdout=subprocess.PIPE)
p.wait()
data = p.stdout.read()
print(heatmap)