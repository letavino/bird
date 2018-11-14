import requests
r = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/List_of_countries_by_GDP_(nominal)")
page = r.json()
print(page["extract"])