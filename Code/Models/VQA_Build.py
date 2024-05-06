#%%
import os
import tensorrt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
from datasets import Dataset
from transformers import (AutoModel, BitsAndBytesConfig, 
                          ViTModel, ViTFeatureExtractor, 
                          AutoModelForCausalLM, AutoTokenizer, AutoModel, 
                          Trainer, TrainingArguments)
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

device = torch.device("cuda")

# %%
DECODER_MODEL = 'D:/HuggingFace/models/MetaAI/Llama_1_13B_Instruct'
TEXT_ENCODER_MODEL = 'distilbert-base-uncased'
IMAGE_ENCODER_MODEL = 'D:/HuggingFace/models/google/vit-base-patch16-224'
# %%
decoder_tokenizer = AutoTokenizer.from_pretrained(DECODER_MODEL)
decoder_tokenizer.pad_token = decoder_tokenizer.eos_token
decoder_tokenizer.padding_side = "right"
image_feature_extractor = ViTFeatureExtractor.from_pretrained(IMAGE_ENCODER_MODEL)

# %%
class MultiModalModel(nn.Module):
    """
    A MultiModal class used to perform visual question answering (VQA).
    It consists of encoders for text and image and a decoder for generating the answer.

    Attributes:
        text_encoder: A model to encode text input.
        image_encoder: A model to encode image input.
        decoder: A model to decode and generate answers.
        text_projection: A linear layer to project text encoding to a specific size.
        image_projection: A linear layer to project image encoding to a specific size.
    """

    def __init__(self, text_encoder_model, image_encoder_model, decoder_model, freeze = None, load_from = None):
        """
        Initialize the MultiModalModel.
        Parameters:
            text_encoder_model (str): Pre-trained text encoder model name.
            image_encoder_model (str): Pre-trained image encoder model name.
            decoder_model (str): Pre-trained decoder model name.
            freeze (str, optional): Which parts of the model to freeze. Can be 'encoders', 'decoder','all', or specific encoder
            load_from (str, optional): Path to a checkpoint file to load the model.
        """

        super(MultiModalModel, self).__init__()

        compute_dtype = getattr(torch, "float16")

        quant_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = compute_dtype,
            bnb_4bit_use_double_quant = True,
        )

        # Initialize text and image encoders
        self.text_encoder = AutoModel.from_pretrained(text_encoder_model)
        self.image_encoder = ViTModel.from_pretrained(image_encoder_model)

        # Initialize the Llama decoder
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model,
                                                            quantization_config = quant_config,
                                                            device_map = {"": 0}
                                                            )
        self.decoder.config.use_cache = False
        self.decoder.config.pretraining_tp = 1

        # Initialize linear layers for projecting encoded features
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, self.decoder.config.hidden_size)
        self.image_projection = nn.Linear(self.image_encoder.config.hidden_size, self.decoder.config.hidden_size)
        
        # Freeze specified encoders if required or load from a checkpoint
        if load_from:
            self.load_model_checkpoint(load_from)
        else:
            self.freeze(freeze)

    def freeze(self, freeze):
        """
        Freeze specific parts of the model to prevent them from being updated during training.

        Parameters:
        freeze (str): Which parts to freeze. Can be 'encoders', 'decoder', 'all', or 'specific encoder'.
        """

        if not freeze:
            return

        print("Freezing...")
        if freeze in ('encoders', 'all') or 'text_encoder' in freeze:
            print("Freezing text encoder")
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        if freeze in ('encoders', 'all') or 'image_encoder' in freeze:
            print("Freezing image enocder")
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        if freeze in ("decoder", "all"):
            print("Freezing decoder (except for cross attention)")
            for name, param in self.decoder.named_parameters():
                if "crossattention" not in name:
                    param.requires_grad = False

    def load_model_checkpoint(self, path):
        """
        Load the model from a saved checkpoint.
        Parameters:
            path (str): Path to the saved checkpoint.
        """
        checkpoint = torch.load(path)
        checkpoint = {k.replace("module.", ""): v for k,v in checkpoint.items()}
        self.load_state_dict(checkpoint)

    def check_input(self, tensor, tensor_name):
        """
        Check if there are any NaN or infinite values in the input tensor.

        Parameters:
            tensor (torch.Tensor): Input tensor.
            tensor_name (str): Name of the tensor for error logging.
        """
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"NaN or infinite values found in {tensor_name}")

    def encode_text(self, input_text, attention_mask):
        """
        Encode text using the text encoder and project it to a specific size.
        Parameters:
            input_text (torch.Tensor): Input text tensor.
            attention_mask (torch.Tensor): Attention mask for the input text.

        Returns:
            torch.Tensor: Projected text encoding.
        """
        self.check_input(input_text, "input_text")
        text_encoded = self.text_encoder(input_text, attention_mask = attention_mask).last_hidden_state.mean(dim = 1)
        return self.text_projection(text_encoded)
    
    def encode_image(self, input_image):
        """
        Encode image using the image encoder and project it to a specific size.

        Parameters:
            input_image (torch.Tensor): Input image tensor.
        
        Returns:
            torch.Tensor: Projected image encoding.
        """
        self.check_input(input_image, "input_iamge")
        image_encoded = self.image_encoder(input_image).last_hidden_state.mean(dim = 1)
        return self.image_projection(image_encoded)
    
    def forward(self, input_text, input_image, decoder_input_ids, attention_mask, labels = None):
        """
        Forward pass through the model.

        Parameters: 
            input_text (torch.Tensor): Input text tensor.
            input_image (torch.Tensor): Input image tensor.
            decoder_input_ids (torch.TEnsor): Decoder input IDs tensor.
            attention_mask (torch.Tensor): Attention mask for the input text.
            labels (torch.Tensor, optional): Ground truth labels for the target.

        Returns:
            torch.Tensor: Decoder output.
        """

        self.check_input(decoder_input_ids, "decoder_input_ids")

        # Encode text and image
        text_projected = self.encode_text(input_text, attention_mask)
        image_projected = self.encode_image(input_image)

        # Combine encoded features
        combined_features = (text_projected + image_projected) / 2
        if labels is not None:
            labels = torch.where(labels == decoder_tokenizer.pad_token_id, -100, labels)

        # Decode with Llama
        decoder_outputs = self.decoder(
            input_ids = decoder_input_ids,
            labels = labels,
            encoder_hidden_states = combined_features.unsqueeze(1)
        )
        return decoder_outputs
    
    def generate(self, image, questions, max_text_length = 5):
        """
        Generate answers for the given image and list of questions.

        Parameters:
            image (Image): Input image.
            questions (list): List of quiestions related to the image.
            max_text_length (int, optional): Maximum text length for generated answers.

        Returns:
            Image: Input image.
        """
        # Encode text and image
        image = retrieve_image(image)
        image_input = image_feature_extractor(images = [preprocess_image(image)], return_tensors = "pt")
        input_image = image_input['pixel_values']
        image_projected = self.encode_image(input_image)

        for question in questions:
            i = text_tokenizer(question, return_tensors = "pt")
            text_projected = self.encode_text(i['inpuit_ids'], i['attention_mask'])

            # Combine encoded features
            combined_features = (text_projected + image_projected) / 2

            generated_so_far = torch.LongTensor([[decoder_tokenizer.bos_token_id]])
            with torch.no_grad():
                for _ in tqdm(range(max_text_length)):

                    decoder_outputs = self.decoder(
                        input_ids = generated_so_far,
                        encoder_hidden_states = combined_features.unsqueeze(1)
                    )

                    next_token_logits = decoder_outputs.logits[:, -1, :]
                    next_token_probs = F.softmax(next_token_logits, dim = -1)
                    next_token = next_token_logits.argmax(-1)
                    confidence = next_token_probs[0, next_token].item()
                    print("Next toekn:", decoder_tokenizer.decode(next_token), "Confidence:", confidence)
                    generated_so_far = torch.cat((generated_so_far, next_token.unsqueeze(0)), dim = 1)
            print(question, decoder_tokenizer.decode(generated_so_far[0]))

        return image

# %%
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_MODEL)
# %%
model = MultiModalModel(
    image_encoder_model=IMAGE_ENCODER_MODEL, 
    text_encoder_model=TEXT_ENCODER_MODEL,
    decoder_model=DECODER_MODEL,
    freeze='nothing')
# %%
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Assuming you have a `MultiModalModel` instance named `model`
num_trainable_params = count_trainable_parameters(model)
print("Number of trainable parameters:", num_trainable_params)
# %%
