FROM python:3.10


# COPY THE APPLICATION CODE
COPY . /app/

WORKDIR /app

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000 8501

ENV API_URL=http://localhost:8000

CMD [ "sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & sleep 3 && streamlit run app.py --server.port 8501 --server.address 0.0.0.0" ]
