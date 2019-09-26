#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess

scenarios = "~/.local/lib/python3.6/site-packages/gfootball/scenarios/"
project = "~/Projects/Python/Football/scenarios"

count = 0

for scenario in os.listdir(project):
    
    if scenario[-3:] != ".py" or scenario == "update.py": continue
    
    source = os.path.join(project, scenario)
    destination = os.path.join(scenarios, scenario)
    
    if os.path.exists(destination): 
        os.remove(destination)
    
    subprocess.run(["cp", source, destination])
    
    count += 1
    
print("Added {} new senario{}".format(count, "s" if count != 1 else ""))

