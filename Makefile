UI_IMAGE = gradio-ui
UI_PORT = 7860

build-ui:
	docker build -t $(UI_IMAGE) -f ui.Dockerfile .

run-ui:
	docker run -p $(UI_PORT):$(UI_PORT) $(UI_IMAGE)