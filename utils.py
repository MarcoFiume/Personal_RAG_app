import json

def load_settings(path = 'settings.json'):
    with open(path) as f:
        return json.load(f)

def save_settings(settings, path = 'settings.json'):
    with open(path, 'w') as f:
        json.dump(settings, f, indent=4)
