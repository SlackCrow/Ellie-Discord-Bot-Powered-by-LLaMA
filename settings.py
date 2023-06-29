TOKEN = ''
model_name_to_use = ''
model_path_to_use = ''
n_ctx = 8192
base_prompt = {'base': [{"role": "system", "content": ""}]}
single_user_mode = False # Each prompt is unique to a single user instead of several users sharing one prompt
multi_server_mode = True # Each prompt is unique to each server instead of server servers sharing one prompt
assistant_name: str = ''
