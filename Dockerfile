FROM python:3.11.9

WORKDIR /code

COPY ./build/app/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./build/app /code/app

EXPOSE $PORT

CMD uvicorn main:app --host 0.0.0.0 --port $PORT