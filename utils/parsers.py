import re
from lxml import etree
import codecs

def parse_en(corpus):
	return re.split('==>.*?<==', corpus)[1:]

def parse_fr(corpus):
	return re.split('<#\d*>', corpus)[1:]

def parse_es(corpus):
	cleaned = re.sub('</doc>|ENDOFARTICLE.', '', corpus)
	return re.split('<.*?>', cleaned)[1:]

if __name__ == '__main__':
	with open('~/dotGit/ml_demo/docs/english/en_corpus.txt') as en_corpus:
		print str(parse_en(en_corpus.read()[:20000])).decode('string_escape')
	print parse_fr("""<#1> This is document 1 <#2> 
		This is document 2 <#3> 
		I am doc 3 <#1241431793>I am another 
		document also <#sdasfd> Not a new document 
		<#6153> This is the last document""")
	with open('~/dotGit/ml_demo/docs/spanish/es_corpus.txt',mode='r') as es_corpus:
		print str(parse_es(es_corpus.read()[:10000])).decode('string_escape')