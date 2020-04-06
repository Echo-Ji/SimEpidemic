import os
import json

DIR_ROOT = 'C:\\Users\\Echo\\Desktop\\nCov\\SimEpidemic\\src\\config'
def read_config():
    with open(os.path.join(DIR_ROOT, 'config.json')) as json_file:
        config = json.load(json_file)
    return config

def update_config(config):
    with open(os.path.join(DIR_ROOT, 'config.json'), 'w') as json_file:
        json.dump(config, json_file, indent=4)
    return

# if __name__ == "__main__":
#     config = read_config()
#     globals().update(config)
#     print(DATA_ROOT)