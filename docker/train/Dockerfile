FROM eidos-service.di.unito.it/eidos-base-pytorch:1.10.0


RUN pip install compressai 
RUN  pip install torchac
RUN pip install ipywidgets
RUN pip install Ninja
RUN pip install pytest-gc

WORKDIR /src
COPY src /src 



RUN chmod 775 /src
RUN chown -R :1337 /src

ENTRYPOINT [ "wandb","agent"]
