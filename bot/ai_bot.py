import os

from decouple import config

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


os.environ['GROQ_API_KEY'] = config('GROQ_API_KEY')


class AIBot:

    def __init__(self):
        self.__chat = ChatGroq(model='llama-3.2-11b-vision-preview')
        self.__retriever = self.__build_retriever()

    def __build_retriever(self):
        persist_directory = '/app/chroma_data'
        embedding = HuggingFaceEmbeddings()

        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
        )
        return vector_store.as_retriever(
            search_kwargs={'k': 30},
        )

    def __build_messages(self, history_messages, question):
        messages = []
        for message in history_messages:
            message_class = HumanMessage if message.get('fromMe') else AIMessage
            messages.append(message_class(content=message.get('body')))
        messages.append(HumanMessage(content=question))
        return messages

    def invoke(self, history_messages, question):
        SYSTEM_TEMPLATE = '''
     
        Você é um Técnico de Suporte especializado em instalações de fibra ótica FTTH da Vodafone Telecomunicações de Portugal. Sua missão é auxiliar outros técnicos que entrarem em contato com dúvidas relacionadas a:

        - Instalações de fibra ótica FTTH, incluindo procedimentos e boas práticas.
        - Diagnóstico e resolução de falhas e problemas no PDO (Ponto de Distribuição Óptico).
        - Configuração de equipamentos de rede, como routers, PLCs, ONUs e modems.
        - Testes e configurações necessárias para garantir conexões de internet estáveis.
        - Sugestões para melhorar instalações e resolver problemas comuns encontrados no processo.
        
        Instruções para Respostas:

        - Responda de forma natural, acolhedora e sempre em português de Portugal.
        - Seja objetivo e claro, oferecendo orientações detalhadas para que o técnico resolva o problema de maneira rápida e eficaz.
        - Mantenha um tom de conversa amigável e colaborativo, como se estivesse orientando um colega técnico.
        - Caso existam diferentes soluções possíveis, explique brevemente cada uma e ajude a escolher a mais adequada.
        - Se precisar de mais detalhes para fornecer uma resposta precisa, solicite as informações necessárias de maneira educada e clara.
        
        Você tambem, caso solicitado, ira criar relatórios técnicos baseado nas informaçoes fornecidas, esses relatórios devem sem tambem objetivo e claro, curto e profissional.
            <context> {context} </context>
        '''

        docs = self.__retriever.invoke(question)
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    'system',
                    SYSTEM_TEMPLATE,
                ),
                MessagesPlaceholder(variable_name='messages'),
            ]
        )
        document_chain = create_stuff_documents_chain(self.__chat, question_answering_prompt)
        response = document_chain.invoke(
            {
                'context': docs,
                'messages': self.__build_messages(history_messages, question),
            }
        )
        return response 