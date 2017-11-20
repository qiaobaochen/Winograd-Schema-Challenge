import requests
from urllib3.util import url

class conceptNet(object):
    """API which cna get knowledge from concept Net"""
    def __init__(self):
        self.url = 'http://api.conceptnet.io/'
        self.lang = 'en'

    def lookup(self, term):
        url_to_search = self.url + "c/en/" + term
        obj = requests.get(url_to_search).json()
        return obj

    def relation(self, term1, term2):
        url_to_search = self.url + "/query?node=/c/en/" + term1 + "&other=/c/en/" + term2
        obj = requests.get(url_to_search).json()
        relations = list()
        for edge in obj['edges']:
            relations.append(edge['rel']['label'])
        return relations
