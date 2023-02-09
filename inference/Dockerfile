FROM python:3.8

WORKDIR /application

COPY ./requirements.txt /application/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /application/requirements.txt

COPY . /application/

RUN chmod a+x start.sh

EXPOSE 8501

CMD ["./start.sh"]
