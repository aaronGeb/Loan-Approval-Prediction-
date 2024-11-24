FROM python:3.11.10-slim
RUN pip install pipenv
WORKDIR /app

COPY Pipfile Pipfile.lock /app/
RUN pipenv install --deploy --system

# Copy prediction.py and loan_model.pkl into the container
COPY scripts/prediction.py /app/
COPY models/loan_model.pkl /app/models/

EXPOSE 9696
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "prediction:app"]