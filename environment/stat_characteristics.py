import os
import json

char_storage_path = './frontend_server/storage/base_the_ville_n25/personas'
char_out_path = './data/characters/'


if __name__ == '__main__':
    char_names = os.listdir(char_storage_path)
    shared_params = {}
    for char_name in char_names:
        char = json.load(open(os.path.join(char_storage_path, char_name, 'bootstrap_memory', 'scratch.json'), 'r', encoding='utf-8'))
        if shared_params == {}:
            shared_params = char.copy()
            continue

        for k, v in char.items():
            if k in shared_params and v != shared_params[k]:
                shared_params.__delitem__(k)

    characteristics = {}
    for char_name in char_names:
        char = json.load(open(os.path.join(char_storage_path, char_name, 'bootstrap_memory', 'scratch.json'), 'r', encoding='utf-8'))

        for k in shared_params:
            char.__delitem__(k)
        characteristics[char_name] = char
    characteristics['shared'] = shared_params
    json.dump(characteristics, open(os.path.join(char_out_path, 'characteristics.json'), 'w', encoding='utf-8'), indent=4)
