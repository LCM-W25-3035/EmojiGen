FROM python:3.11.9

WORKDIR /code

COPY ./build/app/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install fastapi[standard]

COPY ./build/app /code/app
# COPY ./build/app/cgan_emoji_generator.pth /code/app/
# COPY ./build/app/cgan_sticker_generator.pth /code/app/

CMD ["fastapi", "run", "app/main.py", "--port", "80"]