FROM python

ENV PYTHONUNBUFFERED=1

RUN mkdir -p /src && \
    mkdir -p /src/public/static && \
    mkdir -p /src/public/media

WORKDIR /src/app

RUN git clone https://github.com/SolitudeAA11111/BoxCornerIdentifier.git /src/app


RUN pip install -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

WORKDIR /src/app/project


RUN python manage.py makemigrations
RUN python manage.py migrate

RUN  DJANGO_SUPERUSER_PASSWORD=admin python3 manage.py createsuperuser --noinput --username admin --email example@gmail.com
EXPOSE 8000

CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]