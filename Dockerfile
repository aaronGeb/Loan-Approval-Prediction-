FROM python:3.11.10-slim
RUN pip install pipenv
WORKDIR /app

COPY Pipfile Pipfile.lock /app/
RUN pipenv install --deploy --system

COPY scripts/prediction.py models/loan_model.pkl /app/

EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "prediction:app"]