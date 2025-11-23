FROM python:3.11-slim

WORKDIR /app

RUN pip install gradio requests

COPY web_ui.py .

ENV SIM_API_URL="http://backend:8000/simulate"

EXPOSE 7860

CMD ["python3", "web_ui.py"]