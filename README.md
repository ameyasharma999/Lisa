Lisa

1. git clone https://github.com/ameyasharma999/Lisa.git
2. cd Lisa
3. run:
- conda create -n Lisa python=3.9
- conda activate Lisa
- conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
- pip install -r requirements.txt
4. create a folder "checkpoints"
5. Download checkpoints from HERE: https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip
6. Unzip to checkpoints (basespeakers + converter)
7. Install LM Studio (https://lmstudio.ai/)
8. Download Bloke Dolphin Mistral 7B V2 (https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-AWQ) in LM Studio
9. Setup Local Server in LM Studio (go to developer tab, load model and hit "start server")
10. Start Server
11. Get a reference voice in PATH / PATHS (mp3) - Line 217
12. RUN talk.py
13. Follow along to set up the frontend
14. Download and install Blender
15. Download this model
16. Open it in Blender and go to the scripting tab
17. Open blender.py in it
18. Select the model in the viewport CC_Base_Body
19. Run the script    
