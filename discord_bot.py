# bot.py
from header.discord_bot_header import *
from settings import *

client = discord.Client(intents=discord.Intents.all())
gptj = LLM_interface(model_name=model_name_to_use, model_path=model_path_to_use, n_ctx=n_ctx)

block: bool = False

def load_data() -> tuple[dict, list]:
    try:
        f = open('prompt_backup.json')
        prompt = json.load(f)
        f.close()
    except FileNotFoundError:
        prompt = dict()
    try:
        f = open('known_names_backup.json')
        known_names = json.load(f)
        f.close()
    except FileNotFoundError:
        known_names = list()
    return prompt, known_names

prompt, known_names = load_data()

# TODO: edit it so that it uses numpy arrays from the beginning
# Levenshtein distance 
@jit(nopython=True, fastmath=True)
def check_similarity(str1: str, str2:str) -> float:
    if str1 == str2:
        return 1.0

    len1: int = len(str1)
    len2:int  = len(str2)
    matrix: list = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    matrix_np = np.array(matrix)

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i-1] == str2[j-1]:
                cost = 0
            else:
                cost = 1
            matrix_np[i][j] = min(
                matrix_np[i - 1][j] + 1, # Deletion
                matrix_np[i][j - 1] + 1, # Insertion
                matrix_np[i - 1][j - 1] + cost # Substitution
            )

    max_len: int = max(len1, len2)
    similarity = 1 - (matrix_np[len1][len2] / max_len)
    return similarity

@jit(nopython=True)
def detokenize(words) -> str:
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step4 = ''
    for word in step2.split(' '):
        if word in ['.',',',':',';','?','!','%', '"']:
            step4 = step4 + word
        else:
            step4 = step4 + ' ' + word
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

def chat_complete(name: str, message: str, prompt_history_org: dict, server_id: int) -> tuple[T, str]:
    global block, known_names
    while block: 
        time.sleep(0.1)
    prompt_history_to_return = prompt_history_org.copy()
    prompt_history = dict()
    name_without_number: str = name.split('#')[0]
    if not name_without_number in known_names:
        known_names.append(name_without_number)
    del prompt_history_org
    if single_user_mode:
        try:
            prompt_history_to_return[name]
        except:
            prompt_history_to_return[name] = base_prompt['base'].copy()
        prompt_history_to_return[name].append({"role": "user", "content": "" + name_without_number + "» " + message})
        prompt_history = prompt_history_to_return[name]
    else:
        try:
            prompt_history_to_return[server_id]
        except:
            prompt_history_to_return[server_id] = base_prompt['base'].copy()
        prompt_history_to_return[server_id].append({"role": "user", "content": "" + name_without_number + "» " + message})
        prompt_history = prompt_history_to_return[server_id]
    block = True
    result = gptj.chat_completion(prompt_history, repeat_penalty=1.5, temp=1.0)
    block = False 
    print(result)
    if result['choices'][0]['message']['content'] == "" or result['choices'][0]['message']['content'] == '\u200b':
        prompt_history.pop()
        gptj.reset_seed()
        return chat_complete(name=name, message=message+".", prompt_history_org=prompt_history, server_id=server_id)
    response = result['choices'][0]['message']['content']
    response = process_response(response, user_name = name_without_number, currently_known_names=known_names)
    if single_user_mode:
        prompt_history_to_return[name].append({"role": "assistant", "content":response})
        block = False
        return prompt_history_to_return, response
    else:
        prompt_history_to_return[server_id].append({"role": "assistant", "content":"Ellie» " + response})
        block = False
        return prompt_history_to_return, response

# Processing the response before tokenization
@jit(nopython=True)
def pre_process_response(input: str) -> str:
    input = input.replace(':(', "😔").replace(':)', "😊").replace(':D', "😃").replace(':p', "😛").replace(':/','😕').replace(':O','😮')
    response = input.replace('«', '')
    response_split = response.split('»')
    if len(response_split) > 1:
        response = response_split[1]
    else:
        response = response_split[0]
    if response[0] == '"':
        list_response = list(response) 
        list_response[0] = ''
        response = ''.join(list_response)
    elif response.endswith('"'):
        list_response = list(response) 
        list_response[
            len(list_response) - 1
         ] = ''
        response = ''.join(list_response)
    return response 

# Processing the response after tokenization
@jit(nopython=True)
def post_process_response(tokenized_response: list[str], assistant_name: str, currently_known_names: list[str]) -> str:
    if (check_similarity(tokenized_response[0], assistant_name) > 0.8):
        tokenized_response[0] = ''
    for i, token in enumerate(tokenized_response):
        for name in currently_known_names:
            if (check_similarity(name, token) > 0.8):
                tokenized_response[i] = name
    return tokenized_response

# Processing the response to make outputs from LLM to make a bit more sense 
def process_response(response: str, user_name: str, currently_known_names: list[str]) -> str:
    response = pre_process_response(response)
    tokenized_response = list(nltk.word_tokenize(response))
    tokenized_response = post_process_response(tokenized_response, assistant_name, currently_known_names)
    response = detokenize(tokenized_response)
    return response

def record_logs() -> None:
    with open('prompt_backup.json', 'w') as f:
        json.dump(prompt, f, indent=4)
    with open('known_names_backup.json', 'w') as f:
        json.dump(known_names, f)

sched = BackgroundScheduler(daemon=True)
sched.add_job(record_logs,'interval',minutes=1)
sched.start()

# Runs chat_complete in a non-blocking manner so that the bot stays alive while running chat completion.
async def run_chat_complete(chat_complete: typing.Callable, *args, **kwargs) -> typing.Any:
    func = functools.partial(chat_complete, *args, **kwargs)
    return await client.loop.run_in_executor(None, func)

@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    global prompt

    if message.author == client.user:
        return
    
    if multi_server_mode:
        try:
            server_id = message.channel.guild.id
        except:
            server_id = message.id
    else:
        server_id = 0
    
    if not message.content.startswith('!e'):
        if client.user.mention in message.content:
            message_content = message.content[23:]
            print(message_content)
            prompt, response_touse = await run_chat_complete(chat_complete, name=str(message.author).split('#')[0], message=message_content, prompt_history_org=prompt, server_id=server_id)
            await message.channel.send(str(message.author.mention) + " " + response_touse)
    else:
        if message.content.startswith('!e status'):
            if single_user_mode:
                await message.channel.send(assistant_name + " Running in single user mode")
                await message.channel.send("Context Size: " + str(len(prompt[str(message.author).split('#')[0]])))
            else:
                await message.channel.send("Ellie Running in multi user mode")
        elif message.content.startswith('!e server_id'):
            await message.channel.send(str(server_id))

    





client.run(TOKEN)


