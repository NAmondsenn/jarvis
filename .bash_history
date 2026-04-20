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
