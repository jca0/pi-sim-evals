UI_IMAGE = gradio-ui
UI_PORT = 7860
API_IMAGE = pi-sim-evals-api
API_PORT = 9000

build-ui:
	docker build -t $(UI_IMAGE) -f ui.Dockerfile .

run-ui:
	docker run -p $(UI_PORT):$(UI_PORT) $(UI_IMAGE)


build-api:
	docker build -t $(API_IMAGE) -f api.Dockerfile .

run-api:
	docker run -p $(API_PORT):$(API_PORT) $(API_IMAGE)




