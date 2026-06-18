## Start

git clone `<repo-url>` && cd catalyst-propagation-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        
./run_local.sh
