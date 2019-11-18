FROM python:3.7
WORKDIR /main
COPY main .
RUN pip install pipenv \
  && pipenv install --system --deploy
