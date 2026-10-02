from __future__ import annotations
import re
from google import genai
class BudgetExceeded(RuntimeError): pass
class ModelCallError(RuntimeError): pass
class FrozenGemini:
    def __init__(self,model:str,gen:dict,budget): self.model=model;self.gen=gen;self.budget=budget
    def call(self,prompt:str,purpose:str):
        if self.budget.model_calls>=self.budget.max_calls: raise BudgetExceeded('model-call budget exhausted')
        r=genai.Client(api_key=self.budget.api_key).models.generate_content(model=self.model,contents=prompt,config={'temperature':self.gen['temperature'],'top_p':self.gen['top_p'],'max_output_tokens':self.gen['max_output_tokens'],'seed':self.gen['seed']})
        self.budget.model_calls+=1;u=getattr(r,'usage_metadata',None)
        row={'purpose':purpose,'prompt_tokens':getattr(u,'prompt_token_count',None),'output_tokens':getattr(u,'candidates_token_count',None),'total_tokens':getattr(u,'total_token_count',None)}
        self.budget.token_usage.append(row)
        if self.budget.tokens()>self.budget.max_tokens: raise BudgetExceeded('token budget exhausted')
        text=getattr(r,'text','') or ''
        if not text.strip(): raise ModelCallError('model returned empty text')
        return text,row
def extract_cpp(text:str)->str:
    m=re.search(r'```(?:cpp|c\\+\\+)?\\s*(.*?)```',text,re.S|re.I)
    code=(m.group(1) if m else text).strip()
    if not code: raise ModelCallError('empty generated source')
    return code