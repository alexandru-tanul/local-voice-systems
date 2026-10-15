"""The Local Voice Systems dashboard: a web server on this PC for building datasets, training
GPT-SoVITS voices in WSL, and speaking with them. voice_dashboard.py starts it, and the page it
serves is in static/.

Each module holds one part, and the other modules use its names as module.NAME:

config      config.json, and the folders, ports and addresses that follow from it
server      the web server: the page, and the /api and /v1 addresses it answers
jobs        background work, the activity log, and the GPU rule
wsl         commands in WSL, where GPT-SoVITS runs
models      the GPT and SoVITS models in the GPT-SoVITS folder
datasets    datasets: clips, transcripts and reference clips
asr         speech recognition with faster-whisper
clip_check  Check clips
cutting     cutting long recordings into clips
restore     Restore audio
training    Sync & Prepare, and training
compare     Compare epochs
engine      the voice engine, GPT-SoVITS's api_v2.py, and the lines it generates
system      Mic to Voice's services, Stop Everything, and System Check
voices      Export voice and Import a voice
speech_api  OpenAI-style speech for other programs
"""
