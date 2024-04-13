import os
import torch
import torchaudio
from datetime import datetime
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.utils.generic_utils import get_user_data_dir

# Setup the environment and model
os.environ["COQUI_TOS_AGREED"] = "1"
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
ModelManager().download_model(model_name)
model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=os.path.join(model_path, "model.pth"), vocab_path=os.path.join(model_path, "vocab.json"), eval=True, use_deepspeed=True)
model.cuda()

# Function to read text from a file and split into sentences
#def read_text_and_split_into_sentences(file_path):
#    with open(file_path, 'r', encoding='utf-8') as file:
#        text = file.read()
#    sentences = text.split('.')
#    return [sentence.strip() for sentence in sentences if sentence]

def read_text_and_split_into_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Splitting the text by period to get sentences
    sentences = text.split('.')
    cleaned_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # Creating a list to store chunks of up to 20 sentences each
    sentence_chunks = []
    for i in range(0, len(cleaned_sentences), 20):
        # Append chunks of 20 sentences (or fewer, if less than 20 remain)
        sentence_chunks.append(cleaned_sentences[i:i+20])

    return sentence_chunks


# Create a directory for today's date in the temp directory
def create_date_directory():
    today = datetime.now().strftime("%Y-%m-%d")
    directory_path = f"/content/{today}"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    return directory_path

# Prepare and generate speech
def prepare_and_generate_speech(sentence, directory_path, index):
    # Assume 'examples/reference_audio.wav' is a valid reference audio file path
    reference_audio_path = 'examples/female.wav'
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=reference_audio_path)
    out = model.inference(sentence, "tr", gpt_cond_latent, speaker_embedding)
    audio_output_path = os.path.join(directory_path, f"{index + 1}.wav")
    torchaudio.save(audio_output_path, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    print(f"Saved: {audio_output_path}")

# Process text file and generate speech for each sentence
def process_text_file_and_generate_speech(file_path):
    sentences = read_text_and_split_into_sentences(file_path)
    directory_path = create_date_directory()
    for index, sentence in enumerate(sentences):
        if sentence:
            prepare_and_generate_speech(sentence, directory_path, index)

# Specify the path to your text file
text_file_path = '/content/aa.txt'
process_text_file_and_generate_speech(text_file_path)
