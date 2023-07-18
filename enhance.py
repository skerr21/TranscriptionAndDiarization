import torch
import torchaudio
import os
from speechbrain.pretrained import SpectralMaskEnhancement

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
    run_opts={"device": "cuda:0"},
)

def enhance_wav(file_name):
    # Get the basename of the input file
    
    
    # Construct the output file path
    output_path = f"{file_name}_enhanced.wav"
    file_name = os.path.basename(file_name)
    # Load and add fake batch dimension
    noisy = enhance_model.load_audio(file_name).unsqueeze(0)

    # Add relative length tensor
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.0]))
    

    # Saving enhanced signal on disk
    torchaudio.save(output_path, enhanced.cpu(), 16000)

    # Return the path of the enhanced file
    return output_path
