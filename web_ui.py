import os
import tempfile
import requests
import gradio as gr

SIM_API_URL = "http://localhost:9000/simulate"

def run_simulator(scene, policy):
    payload = {
        "scene": int(scene),
        "policy": policy,
    }

    try:
        response = requests.post(SIM_API_URL, json=payload)
    except Exception as e:
        return f"Request failed: {e}"

    if response.status_code != 200:
        return f"Error from simulator API: {response.status_code} {response.text}"

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(fd, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    return tmp_path

with gr.Blocks(title="DROID Simulator") as demo:
    gr.Markdown("## DROID Simulator")

    with gr.Row():
        scene_input = gr.Dropdown(
            label="Scene",
            choices=[
                ("1 – put the cube in the bowl", 1),
                ("2 – put the can in the mug", 2),
                ("3 – put banana in the bin", 3),
                ("4 – put the meat can on the sugar box", 4),
                ("5 – rearrange cubes to spell 'REX'", 5),
                ("6 – stack all the cubes", 6),
            ],
            value=1,
        )
        policy_input = gr.Radio(
            label="Policy",
            choices=["pi0.5", "pi0"],
            value="pi0.5",
        )    

    run_button = gr.Button("Run simulator")
    video_output = gr.Video(label="Simulation video")

    run_button.click(
        fn=run_simulator,
        inputs=[scene_input, policy_input],
        outputs=video_output,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)