# An error-mitigated photonic QCBM

This repository is associated to (add arXiv link).

The original code for the QCBM was written by Tigran Sedrakyan and Alexia Salavrakos. James Mills wrote the code for photon recycling. 

Eric Bertasi and Marion Fabre optimised the code, improved its readability and the structure of the repository.

# Virtual environment
Create and set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
To exit venv:
```bash
deactivate
```

# Test
To run tests:
```bash
source venv/bin/activate
pytest tests
```