cd ~/jarvis
# Initialize git
git init
git branch -M main
# Create .gitignore first (before any commits)
nano .gitignore
git remote add origin https://github.com/NAmondsenn/jarvis.git
# Create folder structure
mkdir -p voice_assistant tests logs models
# Create Python files
touch voice_assistant/__init__.py
touch voice_assistant/main.py
touch voice_assistant/audio.py
touch voice_assistant/wake_word.py
touch voice_assistant/speech_to_text.py
touch voice_assistant/text_to_speech.py
touch voice_assistant/llm.py
touch voice_assistant/config.py
touch voice_assistant/actions.py
# Create config files
touch config.yaml
touch .env
touch requirements.txt
touch README.md
# Create test files
touch tests/test_audio.py
touch tests/test_integration.py
# Verify everything was created
ls -la
ls voice_assistant/
cd ~
ls -la | grep -E "voice_assistant|tests|logs|models|config.yaml|requirements.txt"
nano requirements.txt
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Still in your venv, run:
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
nano requirements.txt
pip install -r requirements.txt
# Deactivate venv temporarily
deactivate
# Install PortAudio system library
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio
# Reactivate venv
source venv/bin/activate
# Try again
pip install -r requirements.txt
git status
git add .
git commit -m "Initial project structure with dependencies and config"
git push
git config --global user.email namondsen@protonmail.com
git config --global user.name NAmondsenn
git commit -m "Initial project structure with dependencies and config"
git push --set-upstream origin main
git commit -m "Initial project structure with dependencies and config"
git push --set-upstream origin main
git add .
git commit -m "Initial project structure with dependencies and config"
git push --set-upstream origin main
git add .
git commit -m "Initial project structure with dependencies and config"
git push --set-upstream origin main
nano .gitignore
nano voice_assistant/audio.py
