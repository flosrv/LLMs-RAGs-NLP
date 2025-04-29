import argparse
import os, sys, hashlib, re, shutil,ollama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
import pandas as pd
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
import argparse, re
from re import Pattern
from pathlib import Path