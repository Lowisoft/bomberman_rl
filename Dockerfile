FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN apt-get update
RUN apt-get -y install gcc g++
#RUN conda install scipy numpy matplotlib numba
#RUN conda install pytorch torchvision -c pytorch
#RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
#RUN pip install pathfinding pyaml igraph ujson
#RUN conda install pandas
#RUN pip install networkx dill pyastar2d easydict sympy pygame

RUN conda install numpy pytorch torchvision cudatoolkit=11.8 -c pytorch -c nvidia
RUN pip install tqdm pathfinding pyaml pygame wandb

##### MANUALLY ADDED START #####
#RUN pip install wandb
ARG WANDB_API_KEY
ENV WANDB_ENTITY=saiboter
ENV WANDB_API_KEY=${WANDB_API_KEY}
##### MANUALLY ADDED END #####

COPY . .
CMD /bin/bash
