import gradio as gr
import pandas as pd

def preview(files, sd: gr.SelectData):
    return files[sd.index].name

def plot_results(df, y_label, x_label, color_discrete_sequence):
    fig = px.bar(df, y=y_label, x=x_label, range_y=[0, 1], height=300, width=400, text=y_label,\
                 color_discrete_sequence=[color_discrete_sequence])
    fig.update_traces(texttemplate='%{text:.0%}', textposition='auto',textfont_size=20) # formats the text on the bar as a percentage
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)', })

    return fig

def pred_labels():
    data = [['tom', 10], ['nick', 15], ['juli', 14]] 
  
    # Create the pandas DataFrame 
    df = pd.DataFrame(data, columns=['Name', 'Age'])

    return gr.BarPlot(df,x="Name",y='Age')

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # f = gr.File(file_types=["image"], file_count="multiple")
            i = gr.Image()
            # preview_button = gr.Button("Preview")
            text_input = gr.Textbox()
            
            submit_button = gr.Button("Submit")
        with gr.Column():
            bar_plot = gr.BarPlot()
    # i.select(preview, i)
    # preview_button.click(lambda x:x, i, o)
    submit_button.click(pred_labels,outputs=[bar_plot])

if __name__ == "__main__":
    demo.launch()