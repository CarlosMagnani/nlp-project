import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify import apply_features
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import mark_negation

nltk.download('punkt')
nltk.download('movie_reviews')

def extrair_recursos(palavras):
    return nltk.FreqDist(palavras)

pos_fileids = movie_reviews.fileids('pos')
neg_fileids = movie_reviews.fileids('neg')
n = min(len(pos_fileids), len(neg_fileids)) // 2
pos_documents = [(list(movie_reviews.words(fileid)), 'pos') for fileid in random.sample(pos_fileids, n)]
neg_documents = [(list(movie_reviews.words(fileid)), 'neg') for fileid in random.sample(neg_fileids, n)]


documentos = pos_documents + neg_documents
random.shuffle(documentos)


tamanho_teste = int(len(documentos) * 0.2)
conjunto_treinamento = apply_features(extrair_recursos, documentos[tamanho_teste:])
conjunto_teste = apply_features(extrair_recursos, documentos[:tamanho_teste])

sentiment_analyzer = SentimentAnalyzer()
trainer = NaiveBayesClassifier.train
classificador = sentiment_analyzer.train(trainer, conjunto_treinamento)

def analisar_polaridade(frase):
    recursos = extrair_recursos(mark_negation(frase.split()))
    polaridade = classificador.classify(recursos)
    return polaridade

exemplos = ['Este filme é excelente e emocionante!',
            'O enredo é confuso e mal desenvolvido.',
            'A atuação dos atores é incrível!',
            'Os efeitos especiais são impressionantes.',
            'Este filme é uma completa perda de tempo.']
for exemplo in exemplos:
    polaridade2 = analisar_polaridade(exemplo)
    print(f'A polaridade da frase "{exemplo}" é {polaridade2}')