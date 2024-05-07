from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from filbertgpt.language_model import load_gpt_4_turbo
from langchain_text_splitters import CharacterTextSplitter
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_core.prompts import PromptTemplate
from filbertgpt.utils.language import language_dict
from langchain.chains import AnalyzeDocumentChain
from filbertgpt.prompt.common import load_language_suffix

# loader = WebBaseLoader("https://waitbutwhy.com/wait-but-who")
# docs = loader.load()

def load_web_IE_chain(        
        lang_code=list(language_dict.keys())[1],
        memory=None
):
    llm = load_gpt_4_turbo()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        f"Given the new context, refine the existing summary in {language_dict[lang_code]}.\n"
        "The new summary should be concise and to the point, and contain all the important information in the context.\n"
        "And its length should not be longer than 1000."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )

    return AnalyzeDocumentChain( 
        combine_docs_chain=chain, 
        text_splitter=text_splitter
    )

# c = load_web_IE_chain()
# result = c(docs[0].page_content)
# print(result["output_text"])