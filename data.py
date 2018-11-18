#TODO LITERALe/substantivierte Verben (call), Plural (scientists), mehr antwortmöglichkeiten

import requests, sys
from termcolor import colored
import numpy as np
from json.decoder import JSONDecodeError
from sklearn.metrics.pairwise import cosine_similarity
from elmo import prepareElmo, compare, getVector, ppd

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

def ranking_elmo(q, vecRef, entity, start, end):
	
	sen = q['label']['value']+" - "+q['desc']['value']#q['item']['value'] , more word entity - >> entity+" - "+
	
	vec = model.predict(ppd(sen, max_length))
	#cos = np.empty(end-start)
	
	arg1 = np.reshape(np.mean(vecRef[0],axis=0),(1,1024))
	arg2 = np.reshape(np.mean(vec[0],axis=0),(1,1024))

	cos = cosine_similarity(arg1, arg2)
	return cos
	
def ranking_2(q):
	return int(q['count']['value'])	

nextLines = 0
sentence = []
max_length=64
model = prepareElmo(max_length)	# Jedes mal neu anpassen?
with open("Conll2003\\train_Q.txt") as f:
	fw = open("Conll2003\\train_Qw.txt", "w", encoding="utf-8")

	lines = f.readlines()
	print("Len Lines:",len(lines))
	for i in range(len(lines)):
		line = lines[i]
		
		if nextLines >0:
			nextLines -=1
			fw.write(line.replace("\n"," ")+" inner\n")
		else:
			n=i
			if line=='\n':
				fw.write(line)
			else:
				fw.write(line.replace("\n"," "))
			sf = line.split(" ")
			#print(line)
			if(line=='\n'):
				sentence = []
			if len(sf)==4:
				sentence.append(sf[0])
				if not sf[1] in ['NN','NNS', 'NNP','NNPS']: #substantiviertes verb call
					fw.write("\n")
				else:	
					entity = sf[0]
					validEntity = ''
					while len(eval(entity, limit=1)) >0:
						validEntity = entity
						validsf = sf
						n=n+1
						line = lines[n]
						sf = line.split(" ")
						if len(sf)==4:
							entity = sf[0]
							entity=validEntity+" "+entity
						else:
							break
					if validEntity == '':
						print("No validEntity")
						fw.write("\n")
					else:
						#print("Found entity", validEntity)
						# fw.write(validEntity+" [")
					
						Q = eval(validEntity)
						
						w = validEntity
						j=0;
						sentenceStart = sentence[0:len(sentence)-2]
						sentenceEnd = []
						m=n
						while lines[m+1] != '\n':	
							sentenceEnd.append(lines[m].split(' ')[0])
							m+=1
						
						sen = sentenceStart+validEntity.split(' ')+sentenceEnd
						#print(colored('hello', 'red'), colored('world', 'green'))
						
						start = len(sentenceStart)
						end =len(sen)-len(sentenceEnd)
						print(sen,"["+str(start)+","+str(end)+"]")
						print(">>>", validEntity)
						refVec= getVector(sen,model,max_length)# Satz!!! mehrere worter in entität
						
						r1a = 0
						r2a = 0

						# TODO: Auf Satz ausbauen
						'''
						for word in lines[max(0,i-5):i]:
							print(word.split(' ')[0])
						print(">>>", validEntity, validsf)
						for word in lines[min(len(lines),i+1):min(len(lines),i+5)]:# nach valid entität
							print(word.split(' ')[0])
						'''
						for j in range(len(Q)):
							
							r1 = ranking_elmo(Q[j], refVec, validEntity,start=start, end=end)
							r2 = int(Q[j]['count']['value'])
							
							tmp = {'value': r1}
							Q[j]['elmo'] = tmp
												
							r1a += r1
							r2a += r2
							
							 #nur relevante Informationen über Q, formatieren
						
						def sort(list):
							newList = []
							max = -1
							sorted = True
							while(sorted):
								sorted = False
								for i in range(len(list)-1):
									for j in range(i+1,len(list)):
										if list[i]['count']['value'] < list[j]['count']['value']:
											tmp = list[i]['count']['value']
											list[i]['count']['value'] = list[j]['count']['value']
											list[j]['count']['value'] = list[i]['count']['value']
											sorted = True
							return list 
							
						Q = sort(Q)
						for j in range(len(Q)):
							print(str(j+1)+")", Q[j]['label']['value'], Q[j]['desc']['value'], Q[j]['elmo']['value'], Q[j]['count']['value'])
							#Q[i][3]  /= r1a
							#Q[i][4]  /= r2a
							
							 
						valid = False	
						while(not valid):
							try: #option: beenden, keine entität, neue entität, unbekannt, (zurück), fehler bei entitätenwahl
								inputNr = int(input())
								valid = True
								if inputNr == 0:
									sys.exit()
								if inputNr > 0 and inputNr <= len(Q):
									fw.write(Q[inputNr-1]['item']['value']+"\n")
									#print(Q[inputNr-1]['label']['value'])
								else:
									print("False number")
									valid = False
								
							except(ValueError):
								print("no number, please try again")
						
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