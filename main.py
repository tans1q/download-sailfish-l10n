import os

import fasttext
import pandas as pd
import requests
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download
from rich.progress import track

DOWNLOAD_FOLDER = "resources"
BASE_URL = "https://translate.sailfishos.org"
model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)


def _get_resource_page_urls(base_page):
    response = requests.get(base_page)
    resources = []
    if response.status_code == 200:
        parser = BeautifulSoup(response.text, 'html.parser')
        for e in parser.select("tbody.stats td.stats-name"):
            resource_name = e.get_text().strip()
            url = e.find('a', href=True)
            if url and resource_name:
                resources.append((resource_name, url['href']))
            else:
                print(f"Cant get resource name or resource url from page")
    else:
        print(f"Error getting resources from {base_page}, status code {response.status_code}")
    return resources


def _get_file_page_urls(resource_urls):
    files = []
    for res_name, res_url in track(resource_urls[:], description="Getting urls of files"):
        res_url = f"{BASE_URL}{res_url}"
        response = requests.get(res_url)
        if response.status_code == 200:
            parser = BeautifulSoup(response.text, 'html.parser')
            for e in parser.select("tbody.stats td.stats-name"):
                file_name = e.get_text().strip()
                url = e.find('a', href=True)
                if url and file_name:
                    files.append((res_name, file_name, url['href']))
                else:
                    print(f"Cant get file name or resource url from page")
        else:
            print(f"Error getting files from {res_url}, status code {response.status_code}")
    return files


def _get_download_urls(files_urls):
    download_urls = []
    for res_name, file_name, file_page_url in track(files_urls[:], description="Getting download links"):
        file_url = f"{BASE_URL}{file_page_url}"
        response = requests.get(file_url)
        if response.status_code == 200:
            parser = BeautifulSoup(response.text, 'html.parser')
            urls = (
                    parser.select("div[id=overview-actions].bd a[title='Download XLIFF file for offline translation']")
                    or
                    parser.select("div[id=overview-actions].bd a[title='Download file in XLIFF format']")
            )
            if not urls:
                print(f"Error getting download link from {file_url}")
                continue

            for e in urls:
                download_urls.append((res_name, file_name, e['href']))
        else:
            print(f"Error getting download link from {file_url}, status code {response.status_code}")
            exit()
    return download_urls


def _download(download_urls):
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    downloaded_files = []
    for res_name, file_name, download_url in track(download_urls[:], description="Downloading files"):
        download_url = f"{BASE_URL}{download_url}"
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            file_name = download_url.split('/')[-1]
            path_to_file = f"{DOWNLOAD_FOLDER}/{file_name}"
            with open(path_to_file, 'wb') as file:
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)
            downloaded_files.append((res_name, file_name, path_to_file))
        else:
            print(f"Error downloading file from {download_url}, status code {response.status_code}")
    return downloaded_files


def _parse(downloaded_files):
    """
    Parse downloaded XML files and return data
    """
    data = []
    for res_name, file_name, path_to_file in track(downloaded_files[:], description="Parsing files"):
        src = f"sailfish/{res_name}/{file_name}"
        with open(path_to_file, 'r') as file:
            parser = BeautifulSoup(file, 'xml')

            if not (elements := parser.select("xliff file body trans-unit[approved='yes']")):
                print(f"No elements found in file {path_to_file}")
                continue

            for e in elements:
                en = e.find('source').get_text().strip()
                tt = e.find('target').get_text().strip()
                if en and tt and _check_is_tatar(tt):
                    data.append({
                        "en": en,
                        "tt": tt,
                        "src": src
                    })
    return data


def _check_is_tatar(text) -> bool:
    """
    Check if the text is in Tatar language
    :param text
    :return: True if the text is in Tatar language, False otherwise
    """
    prediction = model.predict(text.replace("\n", " "))
    return prediction[0][0] == "__label__tat_Cyrl"

def _get_dataframe():
    if os.path.exists("sailfish-tt-en.parquet"):
        return pd.read_parquet("sailfish-tt-en.parquet")

    rpu = _get_resource_page_urls(f"{BASE_URL}/tt")
    print(f"Found {len(rpu)} resources")

    fpu = _get_file_page_urls(rpu)
    print(f"Found {len(fpu)} files to download")

    download_urls = _get_download_urls(fpu)
    print(f"Found {len(download_urls)} download links")

    downloaded_files = _download(download_urls)
    print(f"Downloaded {len(downloaded_files)} files")

    data = _parse(downloaded_files)
    print(f"Found {len(data)} keys")

    df = pd.DataFrame(data)
    df.to_parquet("sailfish-tt-en.parquet", index=False)

    return df


def _merge_with_existing_data(df):
    download_data = hf_hub_download(repo_id="neurotatarlar/tt-en-language-corpus", filename="tt-en.parquet", repo_type="dataset")
    existing_data = pd.read_parquet(download_data)

    # merge and deduplicate two dataframes based on the 'en' and 'tt' columns
    merged = pd.concat([df, existing_data]).drop_duplicates(subset=["en", "tt"])
    return merged


def main():
    df = _get_dataframe()
    merged = _merge_with_existing_data(df)
    merged.to_parquet("tt-en.parquet", index=False)


if __name__ == '__main__':
    main()
