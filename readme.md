To get the timm package to work with the sensor embeddings, you need to install the custom version included in this repo. To do so do cd pytorch-image-models pip install -e .

To use the LLama-Adapter package: Train LlamaAdapter: ./exps/finetune.sh models/llama LLaMA-Adapter/ckpts/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth exps/finetune-data-config.yaml outputs ./exps/finetune.sh models/llama LLaMA-Adapter/ckpts/d26d107eec32127ac86ef1997cf7169de1c56a59c539fc1258c6798b969e289c_LORA-BIAS-7B-v21.pth exps/finetune-data-config.yaml outputs Follow the instructions here: https://github.com/OpenGVLab/LLaMA-Adapter/blob/main/llama_adapter_v2_multimodal7b/docs/train.md You need the llama weights, the adapter weights, and update the config files

