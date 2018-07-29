#FROM debian:jessie
FROM pytorch/pytorch

RUN echo "deb http://deb.debian.org/debian jessie-backports main" >> /etc/apt/sources.list

RUN apt-get -y update && apt-get upgrade -y
RUN apt-get install -y python3 python3-dev python3-pip

RUN apt-get install -y vim build-essential python3-tk x11-apps

RUN mkdir /shared

COPY . /shared/

RUN python3 -m pip install pip setuptools virtualenv --upgrade

RUN mkdir -p /mnt/apps/ && \
    virtualenv -p $(which python3) /mnt/apps/pytorch && \
    echo "source /mnt/apps/pytorch/bin/activate" >> /root/.bashrc && \
    /mnt/apps/pytorch/bin/pip install pip setuptools --upgrade

#RUN cd /shared && /mnt/apps/pytorch/bin/pip install http://download.pytorch.org/whl/cu91/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
RUN cd /shared && /mnt/apps/pytorch/bin/pip install --quiet -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT /entrypoint.sh

ENV DISPLAY :0

VOLUME ["/shared"]
WORKDIR /shared