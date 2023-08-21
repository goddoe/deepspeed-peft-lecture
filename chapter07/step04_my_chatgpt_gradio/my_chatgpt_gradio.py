import gradio as gr
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

model_path = "./ckpts/my_chatgpt_3.8b/checkpoint-52785"
config = PeftConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

inference_model = PeftModel.from_pretrained(model, model_path)
inference_model.eval()

pipe = pipeline(
    "text-generation",
    model=inference_model,
    tokenizer=tokenizer,
    device=0,
    torch_dtype=torch.bfloat16
)

def build_prompt(history):
    conv_list = []
    for user_msg, bot_msg in history: # [[user1, bot1], [user2, bot2], ..., [user10, None]]

        user_turn = f"### User\n{user_msg}"
        conv_list.append(user_turn)

        bot_turn = f"### Bot\n{bot_msg if bot_msg is not None else ''}"
        conv_list.append(bot_turn)
    prompt = "\n\n".join(conv_list)

    print("*")
    print(prompt)
    print("*")
            
    return prompt


def ask(history):
    result = pipe(build_prompt(history),
                   return_full_text=False,
                   # do_sample=True,
                   # top_p=0.3,
                   # repetition_penalty=1.2
                   max_new_tokens=128,
                   )
    return result[0]['generated_text'].split("\n\n")[0]


def user(message, history):
    return "", history + [[message, None]]  # [[user1, bot1], [user2, bot2], ..., [user10, bot10]]


def bot(history):
    bot_msg = ask(history)

    user_msg = history[-1][0]
    history[-1] = [user_msg, bot_msg]
    print("-"*30)
    print(history)
    return history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name="0.0.0.0", server_port=11075)
