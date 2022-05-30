import requests
from bs4 import BeautifulSoup

url = 'https://www.amazon.in/Echo-Dot-3rd-Gen/product-reviews/B07PFFMP9P/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'



def get_url(url):
    r = requests.get(url)
    htmlContent = r.content
    soup = BeautifulSoup(htmlContent, 'html.parser')
    return soup


reviewlist =[]
def get_reviews(soup):
    reviews = soup.find_all('div', {'data-hook':'review'})
        #print(reviews)
    try:
        for item in reviews:
            review={
            'product_name': soup.title.text.replace('Amazon.in:Customer reviews:','').strip(),
            'title' : item.find('a',{'data-hook':'review-title'}).text.strip(),
            'stars' : float(item.find('i',{'data-hook':'review-star-rating'}).text.replace('out of 5 stars','').strip()),
            'body' : item.find('span',{'data-hook':'review-body'}).text.strip(),
            }
            reviewlist.append(review)
    except:
        pass


for i in range(1,1000):
    soup = get_url(f'https://www.amazon.in/Echo-Dot-3rd-Gen/product-reviews/B07PFFMP9P/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber={i}')
    print(f'Getting page: {i}')
    get_reviews(soup)
    print(len(reviewlist))
    if not soup.find('li',{'class':'a-disabled a-last'}):
        pass
    else:
        break

'''for i in range(1,600):
    soup = get_url(f'https://www.amazon.in/Echo-Dot-3rd-Gen/product-reviews/B07PFFMP9P/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber={i}')
    print(f'Getting page: {i}')
    get_reviews(soup)
    print(len(reviewlist))'''

#exporting the data
import pandas as pd

df = pd.DataFrame(reviewlist)
df.to_csv('alexa_reviews.csv',index=False)
print("Finish")