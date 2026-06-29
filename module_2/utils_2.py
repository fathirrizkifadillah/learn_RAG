import requests
import json
import os
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from ipywidgets import Text, Button, VBox, Output, Layout, HTML
from IPython.display import display, clear_output,display, HTML
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from adjustText import adjust_text
from io import BytesIO
import base64
from groq import Groq

def get_proxy_url():
    """
    Get the Groq proxy URL or default endpoint.
    """
    base_url = os.environ.get('GROQ_BASE_URL', 'https://api.groq.com/openai/v1').rstrip('/')
    if "api.groq.com" in base_url and not base_url.endswith("openai/v1"):
        base_url = f"{base_url}/openai/v1"
    return base_url

def get_proxy_headers():
    """
    Get appropriate headers for API calls.
    """
    return {"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}"}

def get_groq_key():
    """
    Get the API key (mapped to Groq API key).
    """
    return os.environ.get("GROQ_API_KEY", "")

def plot_vectors():
    # Define vectors
    v1 = np.array([1, 2])
    v2 = np.array([1, 1])
    array_v = np.array([[3, 2], [5, 6]])

    # Function to calculate cosine similarity
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    # Function to calculate Euclidean distance
    def euclidean_distance(vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    # Calculate cosine similarities and Euclidean distances
    cos_sim_v1_array_v = [cosine_similarity(v1, av) for av in array_v]
    cos_sim_v2_array_v = [cosine_similarity(v2, av) for av in array_v]
    euc_dist_v1_array_v = [euclidean_distance(v1, av) for av in array_v]
    euc_dist_v2_array_v = [euclidean_distance(v2, av) for av in array_v]

    # Create a single plot for both v1 and v2 with array_v
    plt.figure(figsize=(8, 8))

    # Plot v1 and label
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='#fb5c6b')
    plt.text(v1[0] + 0.1, v1[1], f'v1: {tuple(int(x) for x in v1)}', fontsize=9, color='#191c24')

    # Plot v2 and label
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='#fb5c6b')
    plt.text(v2[0] + 0.1, v2[1], f'v2: {tuple(int(x) for x in v2)}', fontsize=9, color='#191c24')

    # Plot array_v vectors and labels
    for i, av in enumerate(array_v):
        plt.quiver(0, 0, av[0], av[1], angles='xy', scale_units='xy', scale=1, color='#08b4cc')
        plt.text(av[0] + 0.1, av[1], f' av[{i}]: {tuple(int(x) for x in av)}', fontsize=9, color='#191c24')

    # Display cosine similarity and Euclidean distance for both v1 and v2 with av
    y_start = 6.5  # Starting y position
    step = 0.3     # Step to separate each line of text
    for i in range(len(array_v)):
        plt.text(0.5, y_start - (2 * i * step), f'Cos(v1, av[{i}]) = {cos_sim_v1_array_v[i]:.4f}', color='#191c24')
        plt.text(0.5, y_start - ((2 * i + 1) * step), f'Dist(v1, av[{i}]) = {euc_dist_v1_array_v[i]:.4f}', color='#191c24')
        plt.text(3.5, y_start - (2 * i * step), f'Cos(v2, av[{i}]) = {cos_sim_v2_array_v[i]:.4f}', color='#191c24')
        plt.text(3.5, y_start - ((2 * i + 1) * step), f'Dist(v2, av[{i}]) = {euc_dist_v2_array_v[i]:.4f}', color='#191c24')

    plt.scatter(array_v[:, 0], array_v[:, 1], color='#191c24', s=10)
    plt.scatter([v1[0], v2[0]], [v1[1], v2[1]], color='#191c24', s=10)

    plt.xlim(0, 6)
    plt.ylim(0, 7)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title("Cosine Similarity & Euclidean Distance", color="#191c24")
    plt.grid(True)
    plt.tight_layout()

    plt.show()



def display_widget(model):
    # Initial data
    sentences = [
        'apple', 'king', 'queen', 'cellphone', 'car', 'automobile', 'fruit', 'man', 'woman',
        "He spoke softly in class", "He whispered quietly during class", "Her daughter brightened the gloomy day"
    ]
    embeddings = model.encode(sentences)

    # PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Output widget for plots
    output_plot = Output()

    # Define pastel colors
    pastel_colors = ['#fa5c69', '#425469', '#00b1cd', '#f44549', '#b3a7d4', '#cccccc', '#8bd3dd', '#c7d3d4', '#738578']

    def plot_embeddings():
        with output_plot:
            clear_output(wait=True)
            plt.figure(figsize=(12, 8))

            # Update colormap for current entries
            color_count = len(sentences)
            colormap = ListedColormap((pastel_colors * (color_count // len(pastel_colors) + 1))[:color_count])

            texts = []
            for color, (label, (x, y)) in zip(colormap.colors, zip(sentences, embeddings_2d)):
                plt.scatter(x, y, color=color, s=100)
                texts.append(plt.text(x, y, label, fontsize=9, color='black'))

            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))

            plt.title('PCA of Word Embeddings', fontsize=16)
            plt.xlabel('Principal Component 1', fontsize=12)
            plt.ylabel('Principal Component 2', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()

    def on_add_word(change):
        nonlocal embeddings_2d, sentences
        new_word = word_input.value.strip()
        if new_word:
            new_embedding = model.encode([new_word])
            new_embedding_2d = pca.transform(new_embedding)

            # Update list and array
            sentences.append(new_word)
            embeddings_2d = np.vstack([embeddings_2d, new_embedding_2d])

            word_input.value = ''  # Clear input
            plot_embeddings()      # Update plot

    # Custom CSS for pastel-colored button
    display(HTML("""
    <style>
        .custom-button {
            background-color: #fa5c69;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 4px;
        }
        .custom-button:hover {
            background-color: #6b252b;
        }
        .output-plot {
            display: flex;
            justify-content: center;
        }
    </style>
    """))

    # Setup the widgets
    word_input = Text(description="Word/Sentence:", layout=Layout(width='400px'), style={'description_width': 'initial'})
    add_button = Button(description="Add Word or Sentence!", layout=Layout(width='200px'))
    add_button.add_class("custom-button")
    add_button.on_click(on_add_word)

    # Display widgets and initial plot
    display(VBox([word_input, add_button], layout=Layout(margin='10px 0')))
    display(HTML("<div class='output-plot'>"))
    display(VBox([output_plot], layout=Layout(align_items='center', justify_content='center')))
    display(HTML("</div>"))

    # Initial plot display
    plot_embeddings()

def map_model(model_name: str) -> str:
    """
    Map Together AI models or other models to appropriate Groq models.
    """
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "openai/gpt-oss-120b"
    ]
    if model_name in groq_models or "groq" in model_name.lower():
        return model_name
    # Fallback to a high-performance Groq model
    return "openai/gpt-oss-120b"

def generate_with_single_input(prompt: str,
                               role: str = 'user',
                               top_p: float = None,
                               temperature: float = None,
                               max_tokens: int = 500,
                               model: str = "openai/gpt-oss-120b",
                               together_api_key = None,
                               groq_api_key = None,
                               **kwargs):
    """
    Call Groq API with a single text prompt.
    """
    # Map model if necessary
    model = map_model(model)

    # Determine the API key to use
    api_key = groq_api_key or together_api_key or os.environ.get("GROQ_API_KEY", "")

    # Check for valid API key when running locally
    if (not api_key or api_key == "your_groq_api_key_here") and "GROQ_BASE_URL" not in os.environ:
        raise ValueError(
            "Groq API key is missing or set to the placeholder ('your_groq_api_key_here').\n"
            "Please obtain a free API key by signing up at https://console.groq.com/,\n"
            "then open the `.env` file in this directory and replace 'your_groq_api_key_here' with your real API key.\n"
            "After updating `.env`, restart your Jupyter notebook kernel and run the cells again."
        )

    # Remove None parameters for Groq API
    payload = {
        "model": model,
        "messages": [{'role': role, 'content': prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    # Remove any Groq AI specific reasoning parameter if present
    kwargs.pop("reasoning", None)

    # Add other kwargs
    payload.update(kwargs)

    # Call Groq API
    if "GROQ_BASE_URL" in os.environ:
        # Use requests/proxy fallback if GROQ_BASE_URL is set
        base_url = os.environ["GROQ_BASE_URL"].rstrip('/')
        if "api.groq.com" in base_url and not base_url.endswith("openai/v1"):
            base_url = f"{base_url}/openai/v1"
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.post(url, json=payload, headers=headers, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}")
    else:
        # Use the official Groq client
        client = Groq(api_key=api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()

    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict


def generate_with_multiple_input(messages: List[Dict],
                                 top_p: float = None,
                                 temperature: float = None,
                                 max_tokens: int = 500,
                                 model: str = "openai/gpt-oss-120b",
                                 together_api_key = None,
                                 groq_api_key = None,
                                 **kwargs):
    """
    Call Groq API with multiple conversation messages.
    """
    # Map model if necessary
    model = map_model(model)

    # Determine the API key to use
    api_key = groq_api_key or together_api_key or os.environ.get("GROQ_API_KEY", "")

    # Check for valid API key when running locally
    if (not api_key or api_key == "your_groq_api_key_here") and "GROQ_BASE_URL" not in os.environ:
        raise ValueError(
            "Groq API key is missing or set to the placeholder ('your_groq_api_key_here').\n"
            "Please obtain a free API key by signing up at https://console.groq.com/,\n"
            "then open the `.env` file in this directory and replace 'your_groq_api_key_here' with your real API key.\n"
            "After updating `.env`, restart your Jupyter notebook kernel and run the cells again."
        )

    # Remove None parameters for Groq API
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p

    # Remove any Together AI specific reasoning parameter if present
    kwargs.pop("reasoning", None)

    # Add other kwargs
    payload.update(kwargs)

    # Call Groq API
    if "GROQ_BASE_URL" in os.environ:
        # Use requests/proxy fallback if GROQ_BASE_URL is set
        base_url = os.environ["GROQ_BASE_URL"].rstrip('/')
        if "api.groq.com" in base_url and not base_url.endswith("openai/v1"):
            base_url = f"{base_url}/openai/v1"
        url = f"{base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.post(url, json=payload, headers=headers, verify=False)
        if not response.ok:
            raise Exception(f"Error while calling LLM: {response.text}")
        try:
            json_dict = json.loads(response.text)
        except Exception as e:
            raise Exception(f"Failed to get correct output from LLM call.\nException: {e}\nResponse: {response.text}")
    else:
        # Use the official Groq client
        client = Groq(api_key=api_key)
        json_dict = client.chat.completions.create(**payload).model_dump()

    try:
        output_dict = {'role': json_dict['choices'][-1]['message']['role'], 'content': json_dict['choices'][-1]['message']['content']}
    except Exception as e:
        raise Exception(f"Failed to get correct output dict. Please try again. Error: {e}")
    return output_dict
