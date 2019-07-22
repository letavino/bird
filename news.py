from iex import Stock
import datetime
import json, os

for i in range(1):
    f = Stock("DIA").price()
    print(f)
    
import webhoseio

def query(n, time):
    timestamp = time-datetime.timedelta(hours=3) #GMT+3 to UTC
    
    webhoseio.config(token="1714afeb-470f-48e4-8c1e-5bba068aa2b1")
    '''
    query_params = {
	"q": "'Volkswagen' language:english site_type:news (published:>"+str(int(timestamp.timestamp())*1000)+" AND published:<+"+str(int((timestamp+datetime.timedelta(days=1)).timestamp())*1000)+")",
	"sort": "relevancy",
    "from": str(n)
    }
    '''
    query_params = {
	"q": " Volkswagen language:english (published:>"+str(int(timestamp.timestamp())*1000)+" AND published:<+"+str(int((timestamp+datetime.timedelta(days=1)).timestamp())*1000)+")",
	"sort": "relevancy",
    "from": str(n)
    }
    output = webhoseio.query("filterWebContent", query_params)
    if not os.path.exists(time.strftime("%Y-%m-%d")):
        os.mkdir(time.strftime("%Y-%m-%d"))
    with open(time.strftime("%Y-%m-%d")+'/testdata_'+str(n)+' '+time.strftime("%Y-%m-%d")+'.json','w') as f:
        json.dump(output, f, indent=2)
    return output['moreResultsAvailable']
        
def getNews():
    time = datetime.datetime(2018, 9, 30)
    n=0
    while query(n, time) > 0:
        n = n+100
    
    '''
    with open('testdata.json','r') as f:
        output = json.load(f)
    print(list(output))
    for i in range(len(output['posts'])):
        print("date:", output['posts'][i]['published'], output['posts'][i]['thread']['country'], output['posts'][i]['thread']['site_type'])
        #print("title:", output['posts'][i]['title']) # Print the text of the first post
        print("text:", output['posts'][i]['text'][:200]) # Print the text of the first post publication date
    #['posts', 'totalResults', 'moreResultsAvailable', 'next', 'requestsLeft']
    '''
# Get the next batch of posts

    #output = webhoseio.get_next()
    #print output['posts'][0]['thread']['site']
    
# Print the site of the first post

getNews()

