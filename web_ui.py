import os
import tempfile
import requests
import gradio as gr

SIM_API_URL = "http://localhost:9000/simulate"

def run_simulator(instruction, episode_length, scene, policy):
    payload = {
        "instruction": instruction,
        "episode_length": float(episode_length),
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
        episode_length_input = gr.Slider(label="Episode length", minimum=1.0, maximum=60.0, step=1.0, value=30.0)
        instruction_input = gr.Textbox(label="Instruction", placeholder="e.g. 'put the cube in the bowl'")
        scene_input = gr.Dropdown(
            label="Scene",
            choices=[
                ("1 – cube and bowl", 1),
                ("2 – can and mug", 2),
                ("3 – banana and bin", 3),
                ("4 – cluttered scene", 4),
                ("5 – alphabet cubes", 5),
                ("6 – colored cubes", 6),
            ],
            value=1,
        )
        policy_input = gr.Dropdown(
            label="Policy",
            choices=["pi0.5", "pi0"],
            value="pi0.5",
        )    

    with gr.Row():
        run_button = gr.Button("Run simulator", variant="stop")
        stop_button = gr.Button("Stop simulator")
    
    video_output = gr.Video(label="Simulation video")

    run_event = run_button.click(
        fn=run_simulator,
        inputs=[instruction_input, episode_length_input, scene_input, policy_input],
        outputs=video_output,
    )

    stop_button.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[run_event]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)