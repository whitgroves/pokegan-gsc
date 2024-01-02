import os
import re
import requests
from bs4 import BeautifulSoup

data_dir = './.data/'
assert os.path.isdir(data_dir)

# Thanks to https://www.dragonflycave.com/resources/pokemon-list-generator
listfile = os.path.join(data_dir, 'pokedex.txt')
assert os.path.isfile(listfile)
with open(listfile, encoding='utf-8') as infile:
    pokedex = [line.strip() for line in infile.readlines()]

for pokemon in pokedex:
    pokemon = pokemon.replace(' ', '_')
    pokemon = pokemon.replace('’', '%27')
    pokemon = re.sub('([^\w|\s|/.|%27|-])', '', pokemon) # it had to be like this
    try:
        url = f'https://bulbapedia.bulbagarden.net/wiki/File:{pokemon}.png'
        soup = BeautifulSoup(requests.get(url).content, 'html.parser')
        src = soup.find('img').attrs.get('src') # first image is usually the pokemon
        image_file = requests.get(f'http:{src}', stream=True)
        with open(os.path.join(data_dir, 'pokedex', f'{pokemon}.png'), 'wb') as outfile:
            outfile.write(image_file.content)
    except Exception as e:
        print(e)
        break
        # print(f'could not get image for pokemon: {pokemon}')
