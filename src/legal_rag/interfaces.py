from .pipeline import OVERTIME_CASE


def render(result):
    text = result["answer"]
    if result.get("sources"):
        text += "\n\n**Sumber:**\n" + "\n".join(f"- [{s['id']}] {s['citation']}" for s in result["sources"])
    return text


def interactive_loop(rag):
    """Required notebook interface. input() and Markdown are deliberately explicit."""
    from IPython.display import display, Markdown
    while True:
        question = input("Pertanyaan legal (ketik keluar): ").strip()
        if question.lower() in {"keluar", "exit", "quit"}:
            break
        if question:
            display(Markdown(render(rag.ask(question))))


def launch_gradio(rag):
    import gradio as gr
    def respond(question):
        return render(rag.ask(question))
    with gr.Blocks(title="LexID · Legal Research Assistant") as demo:
        gr.Markdown("# LexID\nAsisten riset tim legal · sumber dapat diperiksa\n\nKorpus pembelajaran historis; verifikasi status regulasi sebelum digunakan.")
        question = gr.Textbox(label="Pertanyaan", lines=3, value=OVERTIME_CASE)
        button = gr.Button("Cari dasar hukum", variant="primary")
        answer = gr.Markdown()
        button.click(respond, question, answer)
    return demo.launch(server_name="127.0.0.1", share=False)
