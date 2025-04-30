brew install libomp
brew --version
python3 pofa_concrete_prediction.py
eval "$(/opt/homebrew/bin/brew shellenv)"

pip install -r requirements.txt
python3 -m venv venv => starts a virtual enviromnent
source venv/bin/activate => activate the virtual environmnet
