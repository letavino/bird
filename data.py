import requests
from json.decoder import JSONDecodeError

def eval(entity, limit=10000):
	#print("search:",entity)
	try:
		query = '''
			SELECT ?item ?desc ?article
			WHERE 
			{
				VALUES ?name { "'''+entity+'''"@en "'''+entity.lower()+'''"@en "'''+entity.upper()+'''"@en "'''+entity.capitalize()+'''"@en } #camal case? other posibilities Untersuchung: Abweichung vom Geschrieben zum Label
				?item skos:altLabel|rdfs:label "'''+entity+'''"@en;
				rdfs:label ?label;
				schema:description ?desc .
				FILTER (lang(?label)="en")
				FILTER (lang(?desc)="en")
				OPTIONAL {
					?article schema:about ?item .
					?article schema:inLanguage "en" .
					?article schema:isPartOf <https://en.wikipedia.org/> .
				}
			} LIMIT '''+str(limit)+'''
			'''
		#print(query)
		url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
		data = requests.get(url, params={'query': query, 'format': 'json'}).json()
	except JSONDecodeError:
		#print("Not found")
		return([])
		
	return data['results']['bindings']
	

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
					candidates = eval(validEntity)
					for candidate in candidates:
						fw.write(" candidate [")
						for field in candidate:
							fw.write(candidate[field]['value']+" ")
						fw.write("]")
					fw.write("]\n")
					nextLines = n-i-1

#rank by usage or other statistics
#the european instead of european commision