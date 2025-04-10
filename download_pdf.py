import requests
from bs4 import BeautifulSoup
import os
import time

def get_arxiv_ids(key_word, order_type, max_papers=6):
    """
    Fetches arXiv paper IDs from the search results page.
    :return: List of arXiv paper IDs
    """

    num_page = 0
    arxiv_ids = []

    while num_page <= int(max_papers/25):
      num_page=num_page+1
      url = f"https://arxiv.org/search/?searchtype=all&query={key_word.replace(' ','+')}&abstracts=show&size=25&order={order_type}&start={num_page}"
      response = requests.get(url)
      if response.status_code != 200:
        print("Failed to fetch search results.")
        return []

      soup = BeautifulSoup(response.text, "html.parser")
      paper_links = soup.find_all("p", class_="list-title is-inline-block")

      for link in paper_links:
        arxiv_id = link.a.text.strip().replace("arXiv:", "")
        arxiv_ids.append(arxiv_id)
        if len(arxiv_ids) >= max_papers:
            break

    return arxiv_ids


def download_arxiv_pdf(arxiv_id, save_path="docs/"):
    """
    Downloads an arXiv PDF given its ID.

    :param arxiv_id: str, arXiv paper ID
    :param save_path: str, Directory to save the PDF
    """
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    response = requests.get(pdf_url, stream=True)
    if response.status_code == 200:
        file_path = os.path.join(save_path, f"{arxiv_id}.pdf")
        with open(file_path, "wb") as pdf_file:
            for chunk in response.iter_content(1024):
                pdf_file.write(chunk)
        print(f"Downloaded: {file_path}")
    else:
        print(f"Failed to download {arxiv_id}. HTTP Status Code: {response.status_code}")


if __name__ == "__main__":
    """
    the parameter <order_type> can take one of these 5 options
    1) announced_date(newest first) : -announced_date_first
    2) announced_date(oldest first) : announced_date_first
    3) submission date(newest first) : -submitted_date
    4) submission date(oldest first) : submitted_date
    5) relevance : ''
    """

    #Example: Download PDFs for "Gen AI" papers
    key_word = "Large Language Model" #Replace at your convenience
    order_type = "" #"-announced_date_first" #Replace at your convenience based of the option define above

    arxiv_ids = get_arxiv_ids(key_word, order_type, max_papers=4)
    print(f"Found {len(arxiv_ids)} papers.")

    for arxiv_id in arxiv_ids:
        download_arxiv_pdf(arxiv_id)
        time.sleep(1)  # Avoid overwhelming the server