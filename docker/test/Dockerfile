


FROM ubuntu:latest
WORKDIR /src
COPY src /src


RUN  apt-get update \
    && apt-get install python3-pip -y

RUN pip install --upgrade pip




RUN pip install compressai 
RUN  pip install torchac
RUN pip install Ninja





RUN chmod 775 /src
RUN chown -R :1337 /src

ENTRYPOINT [ "python3"]
