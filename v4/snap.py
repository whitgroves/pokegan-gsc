import os
import requests
import bs4

def get_images(dex_list:os.PathLike, save_root:os.PathLike) -> None:
    # listings are in the "000_namelower" format, with special characters removed/replaced
    # once again, thanks to https://www.dragonflycave.com/resources/pokemon-list-generator
    with open(dex_list, encoding='utf-8') as file:
        dex = [line.strip().split('_') for line in file.readlines()]
    dex = {int(num): name for num, name in dex}

    # gens
    # 1 = 001-151
    # 2 = 152-251
    # 3 = 252-386
    # 4 = 387-493
    # 5 = 494-649
    # 6 = 650-721
    # 7 = 722-809
    # 8 = 810-905 (not available)
    # 9 = 906-MAX (not available)

    n_images = 0
    base_url = 'https://www.pokencyclopedia.info/' # special thanks
    for num, name in dex.items():
        if num > 807: break # 807 is the last index available
        elif num > 721: gen = 7
        elif num > 649: gen = 6
        elif num > 493: gen = 5
        elif num > 386: gen = 4
        elif num > 251: gen = 3
        elif num > 151: gen = 2
        elif num > 0: gen = 1
        else: raise ValueError('MissingNo.')
        label = f'{num:03} {name}'
        print(f'Fetching images for {label}...'+' '*16, end='\r')
        res = requests.get(f'{base_url}/en/index.php?id=sprites/{num:03}')
        if res.status_code != 200:
            print(f'Skipped fetch for {label}: Bad Response: {res}')
            continue
        soup = bs4.BeautifulSoup(res.content, 'html.parser')
        image_urls = [image.attrs.get('src').replace('..', base_url)
                  for image in soup.find_all('img', attrs={'alt': f'#{num:03}'})]
        image_urls = list(filter(lambda url: url[-4:] == '.png', image_urls)) # no GIFs
        save_dir = os.path.join(save_root, f'gen{gen}')
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        cached = os.listdir(save_dir)
        skip_filter = ['sprf_', 'o-', 'o_', 'md__', 'ico_', 'b_', 'box_', 'mod_'] # back sprites, etc.
        n_skipped = 0
        n_cached = 0
        n_saved = 0
        for url in image_urls:
            file_name = url.split("/")[-1]
            if any(file_name.startswith(x) for x in skip_filter):
                n_skipped += 1
                continue
            if file_name in cached: # waste not, want not
                n_cached += 1
                continue 
            image = requests.get(url, stream=True).content
            with open(f'{save_dir}/{file_name}', 'wb') as file:
                file.write(image)
                n_saved += 1
        print(f'Saved {n_saved} images for {label} in {save_dir} ({n_cached} cached) ({n_skipped} skipped)')
        n_images += n_cached + n_saved
    print(f'Fetch complete. Total images on disk: {n_images}')

if __name__ == '__main__':
    get_images('v4/pokedex.txt', 'v4/.data/')
