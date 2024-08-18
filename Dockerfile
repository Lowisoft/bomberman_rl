FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN apt-get update
RUN apt-get -y install gcc g++
RUN conda install scipy numpy matplotlib numba
# MANUALLY CHANGED:
#RUN conda install pytorch torchvision -c pytorch
RUN conda install pytorch torchvision cudatoolkit=11.8 -c pytorch -c nvidia
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
RUN pip install pathfinding pyaml igraph ujson
RUN conda install pandas
RUN pip install networkx dill pyastar2d easydict sympy pygame
# MANUALLY ADDED START
RUN pip install wandb
ARG WANDB_API_KEY
ENV WANDB_ENTITY=saiboter
ENV WANDB_API_KEY=${WANDB_API_KEY}
# MANUALLY ADDED END
COPY . .
CMD /bin/bash
