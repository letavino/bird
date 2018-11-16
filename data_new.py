import requests, sys
from json.decoder import JSONDecodeError

def eval(entity, limit=10000):
	#print("search:",entity)
	try:
		query = '''
		SELECT ?item ?label ?desc ?article (count(?sitelink) as ?count) WHERE {
			VALUES ?name { "'''+entity+'''"@en "'''+entity.lower()+'''"@en "'''+entity.upper()+'''"@en "'''+entity.capitalize()+'''"@en}
			?item skos:altLabel|rdfs:label ?name ;
			rdfs:label ?label .
			?sitelink schema:about ?item .
			?item schema:description ?desc.
			MINUS {?item wdt:P31 wd:Q4167410 .}
			FILTER (lang(?label) = "en")
			FILTER (lang(?desc) = "en")
			OPTIONAL {
				?article schema:about ?item .
				?article schema:inLanguage "en" .
				?article schema:isPartOf <https://en.wikipedia.org/> .
			}
		} 
		GROUP BY ?item ?label ?desc ?article 
		LIMIT '''+str(limit)+'''
		'''
		#print(query)
		url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
		data = requests.get(url, params={'query': query, 'format': 'json'}).json()
	except JSONDecodeError:
		#print("Not found")
		return([])
		
	return data['results']['bindings']

def ranking_elmo(q):
	return len(q[0]) #todo
def ranking_2(q):
	return int(q['count']['value'])	

nextLines = 0
	
with open("Conll2003\\train_Q.txt") as f:
	fw = open("Conll2003\\train_Qw.txt", "w", encoding="utf-8")

	lines = f.readlines()
	print(len(lines))
	for i in range(len(lines)):
		print(i)
		line = lines[i]
		if nextLines >0:
			nextLines -=1
			fw.write(line.replace("\n"," ")+" inner\n")
		else:
			n=i

			fw.write(line.replace("\n"," "))
			sf = line.split(" ")
			if len(sf)==4:
				entity = sf[0]
				validEntity = ''
				while len(eval(entity, limit=1)) >0:
					validEntity = entity
					n=n+1
					line = lines[n]
					sf = line.split(" ")
					if len(sf)==4:
						entity = sf[0]
						entity=validEntity+" "+entity
					else:
						break
				if validEntity == '':
					fw.write("\n")
				else:
					#print("Found entity", validEntity)
					fw.write(validEntity+" [")
					Q = eval(validEntity)
					
					r1a = 0
					r2a = 0

					for i in range(len(Q)):
						r1 = ranking_1(Q[i])
						r2 = ranking_2(Q[i])
						
						r1a += r1
						r2a += r2
						
						Q[i].append(r1)
						Q[i].append(r2)

					for i in range(len(Q)):
						Q[i][3]  /= r1a
						Q[i][4]  /= r2a
										
					print(lines[max(0,i-5):i])
					print(">>>", validEntity)
					print(lines[min(len(lines),i+1):min(len(lines),i-5)])
					
					for i in range(len(Q)):
						print((i+1), Q[i]) #nur relevantes, formatieren
					
					input = input()

					try:
						n = int(input)

						if n == 0:
							print("Null")
						if n > 0 and n <= len(Q):
							Q[n-1].append(1)
							print(Q[n-1])
						else:
							print("False number")
					except(ValueError):
						print("no number")
					
					sys.exit()
					'''
					for candidate in candidates:
						fw.write(" candidate [")
						for field in candidate:
							fw.write(candidate[field]['value']+" ")
						fw.write("]")
					fw.write("]\n")
					'''
					nextLines = n-i-1

#rank by usage or other statistics
#the european instead of european commision