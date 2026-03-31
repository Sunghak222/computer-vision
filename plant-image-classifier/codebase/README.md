# 0. download necessary libraries
pip install -r requirements.txt

# 1. split
python src/split_dataset.py

# 2. augmentation (only once)
python src/augmentation.py

# 3. train
python src/train.py

# 4. evaluate
python src/evaluate.py

# How to run application
0. Modify MODEL_PATH in src/app.py if you want to try other models.
1. enter "streamlit run src/app.py" in terminal, then you would link to the webpage
2. input an image
3. click the predict button

