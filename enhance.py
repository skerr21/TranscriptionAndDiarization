import torch
import torchaudio
import os
from speechbrain.pretrained import SpectralMaskEnhancement

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
    run_opts={"device": "cuda:0"},
)

def enhance_wav(input_path, enhance=True):
    # Get the basename of the input file
    waveform, sample_rate = torchaudio.load(input_path)

    # Enhance the audio if requested
    if enhance:
        # Load the denoiser model
        file_name = os.path.basename(input_path)
        output_path = f"{file_name}_enhanced.wav"
        noisy = enhance_model.load_audio(file_name).unsqueeze(0)

        # Add relative length tensor
        enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))

        # Saving enhanced signal on disk
        torchaudio.save(output_path, enhanced.cpu(), 16000)
        
        # Return the path of the enhanced file
        return output_path
    else:
        return input_path