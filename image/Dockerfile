FROM python:3.6
ENV VERSION 1
WORKDIR E:\Github\machine_learning\image

COPY ./qrocde /
RUN cd /qrcode \
    &&  pip install -r requirements.txt

CMD [ "python", "web_server.py" ]

EXPOSE 2000