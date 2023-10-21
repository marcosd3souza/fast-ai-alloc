![FastAiAlloc](https://github.com/marcosd3souza/fast-ai-alloc/blob/main/assets/2-RTAI_workflow.jpg)

# FastAiAlloc: A real-time multi-resources allocation framework proposal based on predictive model and multiple optimization strategies

In cloud platforms, a common task is to allocate computational resources, e.g., memory and CPU, requested by applications and users. The allocation of these resources, which is an optimal load-balancing task, is considered an NP-Hard problem, being a challenging research area. There are many works proposed in the literature to address this problem. They use several strategies to deal with this optimization problem such as evolutionary algorithms, exact programming and also heuristics. However, some steps of the allocation process are not considered by these works, which are sometimes treated separately and not in an integrated manner. These steps include the applications and users resource consumption request profile as part of the optimization process and defining adequate metrics to check the optimized allocation. To integrate these steps, this work proposes a framework based on the following strategies, widely used in the literature: Genetic Algorithms (GA), Particle Swarm Optimization (PSO) and Linear Programming, besides our Heuristic approach. Furthermore, we restricted the resource allocation optimization to a real-time scenario, facilitating its use in an industrial process. In addition, Key Performance Indicators (KPIs) are proposed to carry out a comparative study in a public dataset. Our experiments showed that our proposed framework achieved better results than the baseline one, e.g., using a random allocation. In some cases, we observed an increase of almost 60% in performance compared with the baseline. In addition, when trying to balance memory and CPU consumption in the cluster, the linear approach performed the best, while the GA achieved the best result in allocating different user profiles across the cluster.

# Paper

Oliveira, M. D. S., Junior, F. E. F., Cerveny, L., Miyazaki, F. A., Neri, L. V., da Silva, A., ... & de Lima, Ó. D. W. (2023). FastAiAlloc: A real-time multi-resources allocation framework proposal based on predictive model and multiple optimization strategies. Future Generation Computer Systems, 149, 622-636.

```
@article{DESOLIVEIRA2023622,
title = {FastAiAlloc: A real-time multi-resources allocation framework proposal based on predictive model and multiple optimization strategies},
journal = {Future Generation Computer Systems},
volume = {149},
pages = {622-636},
year = {2023},
issn = {0167-739X},
doi = {https://doi.org/10.1016/j.future.2023.08.014},
url = {https://www.sciencedirect.com/science/article/pii/S0167739X23003126},
author = {Marcos {de S. Oliveira} and Francisco Erivaldo Fernandes and Lukas Cerveny and Flávia Akemi Miyazaki and Leonardo Valeriano Neri and Alan {da Silva} and Beatriz Leandro Bonafini and Victor Medeiros Outtes Alves and Órion Darshan Winter {de Lima}},
keywords = {Cloud computing, Resource allocation, Self-Organizing Maps, Exact programming, Particle Swarm Optimization, Genetic Algorithms}
}
```
# Execution
------------------------------------------------------------------------------------------------------------------------------------------------------

step-by-step to execute the experiments


step 1) make sure a newer python version (e.g >=3.8) is installed on your machine by ```python --version```

step 2) install conda, miniconda or other virtual environment manager

step 3) create an environment: ```conda create --name fast-ai-alloc-experiments```

step 4) activate the new created env: ```conda activate fast-ai-alloc-experiments```

step 5) install the requirements located in root: ```pip install -r requirements.txt```

step 6) prompt main.py in terminal: ```python3.8 ./code/main.py```
