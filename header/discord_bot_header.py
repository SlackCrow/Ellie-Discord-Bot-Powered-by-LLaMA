from apscheduler.schedulers.background import BackgroundScheduler
from difflib import SequenceMatcher
from typing import TypeVar
from llm_complete_interface import LLM_interface
from numba import jit
import numpy as np
import functools
import discord
import typing
import nltk
import json
import time
import re

T = TypeVar("T")

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')
