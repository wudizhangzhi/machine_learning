FROM python:3.6
ENV VERSION 1
WORKDIR /

COPY ./qrcode /
RUN cd / \
    &&  pip install -r requirements.txt -i http://pypi.douban.com/simple --trusted-host pypi.douban.com

CMD [ "python", "web_server.py" ]

EXPOSE 2000