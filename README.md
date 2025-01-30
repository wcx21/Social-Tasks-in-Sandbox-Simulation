# Social Tasks in Sandbox Simulation

This repository is for [Towards Objectively Benchmarking Social Intelligence \\for Language Agents at the Action Level](https://arxiv.org/abs/2404.05337). In this project, we try to evaluate the emerging social intelligence of language agents with objective metrics at the action level in simulation environments.
Our code is built on the [Generative Agents](https://github.com/joonspk-research/generative_agents "Generative Agents").  

## Requirements/Setup
1. Please first follow the instructions of [Generative Agents](https://github.com/joonspk-research/generative_agents/commit/fe05a71d3e4ed7d10bf68aa4eda6dd995ec070f4) to configure the environment. 
2. We suggest installing Selenium for effectively run the code effectively on headless servers. We use Chrome driver, though others may also work.
3. Some of our data are zipped in [base_0209.zip](environment%2Ffrontend_server%2Fstorage%2Fbase_0209.zip) and [base_1124.zip](environment%2Ffrontend_server%2Fstorage%2Fbase_1124.zip), unzip them if you want to run our designed tasks at [task_0905](environment%2Fdata%2Ftask_0905).

**Note: We made some modifications on the code of generative agents, the behaviors of some code may be different from the original version.**

## Run simulation with social tasks
The entrance is [reverie_with_task.py](reverie%2Fbackend_server%2Freverie_with_task.py). By default, we enable the tdp module. It can be disabled by using '--naive'.
```shell
python reverie_with_task.py -t ../../environment/data/task_0905/tasks.json -i 0 -m gpt-35-turbo
```
The simulation folder will be automatically generated, while the simulation still needs an order to start, for example:
```
run 2800
```
And it also needs manually saving the results.
```
finish
```

After that, the simulation can be evaluated with [post_run_eval_v3.py](environment%2Fpost_run_eval_v3.py), for example:
```shell
python post_run_eval_v3.py --sim_root ./frontend_server/eval/gpt4-tdp-2
```
Where the *sim_root* is a folder containing multiple simulations.


### Language-level evaluation 
Simulations started by reverie_with_task.py will automatically record task-related conversations and save them under *reverie/backend_server/data/dumped_data*.
We can perform language-level evaluates with [eval_main.py](reverie%2Fbackend_server%2Foffline_eval%2Feval_main.py), for example：
```shell
python offline_eval/eval_main.py -p ./data/offline_eval_full -s ./data/results -m llama2_13b --model_path <path_to_model> --eval --re_run --re_gen
```
**Note: we may need to first collect the dumped conversation.**

## Parallelization
Reproducing our results may need hundreds of simulation runs, thus we suggest running the experiments in parallel. 
Our solution is to run experiments in multiple docker containers, and use a separate process to simulate the behavior of the web explorer.

In our practice, we start 3 separate processes in each container:
```shell
python manage.py runserver
```
```shell
python reverie_with_task.py -t <task_file> -i <task_id> [**params]
# In the python console  
run <n_steps>
```
```shell
python selenium_proc.py
```

The script for collecting results from multiple docker containers depends on the specified environment. Generally, we use something like [collect_l2.py](reverie%2Fbackend_server%2Fcollect_l2.py)

## Reference
If you use our code, please consider citing our paper:
```bibtex
@inproceedings{wang2024towards,
  title={Towards Objectively Benchmarking Social Intelligence for Language Agents at Action Level},
  author={Wang, Chenxu and Dai, Bin and Liu, Huaping and Wang, Baoyuan},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
  pages={8885--8897},
  year={2024}
}
```

Some code and data are from [Generative Agents](https://github.com/joonspk-research/generative_agents "Generative Agents"):
```bibtex
@inproceedings{Park2023GenerativeAgents,  
author = {Park, Joon Sung and O'Brien, Joseph C. and Cai, Carrie J. and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S.},  
title = {Generative Agents: Interactive Simulacra of Human Behavior},  
year = {2023},  
publisher = {Association for Computing Machinery},  
address = {New York, NY, USA},  
booktitle = {In the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)},  
keywords = {Human-AI interaction, agents, generative AI, large language models},  
location = {San Francisco, CA, USA},  
series = {UIST '23}
}
```
