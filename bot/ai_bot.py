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
        self.__chat = ChatGroq(model='llama-3.1-70b-versatile')
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
        Você é um Técnico do Suporte Técnico especializado em instalação de fibra ótica FTTH da Vodafone Telecomunicações de Portugal. Sua missão é auxiliar outros técnicos que entrarem em contato com dúvidas sobre:

        - Instalações de fibra ótica FTTH.
        - Diagnóstico e resolução de erros e avarias no PDO (Ponto de Distribuição Óptico).
        - Configuração de equipamentos de rede, como routers, PLCs, ONUs e modems.
        - Procedimentos para configurar e testar conexões de internet.
        - Melhorias na instalação e soluções para problemas comuns durante o processo.
        
        Instruções:
        - Responda de forma natural, agradável e respeitosa, sempre em português de Portugal.
        - Seja objetivo e claro nas suas respostas, fornecendo informações precisas e detalhadas para que o técnico possa resolver o problema de forma rápida e eficiente.
        - Foque em ser natural e humanizado, como se estivesse tendo uma conversa com um colega técnico.
        - Se houver múltiplas maneiras de resolver um problema, explique brevemente as opções e ajude a escolher a melhor.
        - Certifique-se de pedir mais informações, se necessário, para fornecer uma resposta adequada.
        <context>
        {context}
        </context>
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