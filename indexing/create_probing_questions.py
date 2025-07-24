from openai import OpenAI, AzureOpenAI
from search_document import load_documents, search_by_id
import json
from tqdm import tqdm
import pickle
import sentence_transformers

import os

from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY", "")

def generate_probing_questions(document):
	openai_client = OpenAI(api_key=openai_api_key)

	system_prompt = '''
		instruction = r"""
		You are ProbingQuestionGenerator

		Inputs  
		  1. **context** – a Gilead IT‑support article (English).  

		Goal  
		Return the minimal, non‑redundant set of prerequisite questions the agent must ask *before* troubleshooting, **plus** the expected answers and what to do for each answer.

		────────────────────────────────────────────
		Workflow
		1. **Extract prerequisites** – device/OS, network/VPN, admin rights, auth codes, peripherals, physical condition, required software, etc.

		2. **Cluster & merge** equivalent prerequisites.  
		   •Create one concise question per cluster.  
		   •Skip clusters already satisfied in *optional_user_history*.

		  **Prerequisite‑only scope**  
		   • A prerequisite is something the user must *possess, know, or have access
		     to* (device type, admin rights, VPN, codes, peripherals, etc.).  
		   • **Do NOT include** diagnostics or troubleshooting steps (e.g. “open Task
		     Manager”, “press the power button”, “run Dell Command Update”).  
		   • If the article’s sentence begins with “Press…”, “Run…”, “Click…”, that is a
		     troubleshooting step—exclude it.

		3. **Build the answer map** for every question  

		   ► Decide the **answer type**  
		     •Binary → `"type": "yes_no"`  
		     •Fixed list → `"type": "one_of"`  
		     •Anything else → `"type": "text"`

		   ► For each option provide **next‑step guidance**:  
		     •If the option satisfies the prerequisite → `"next": "proceed"`  
		     •If not, summarise the specific action(s) from the article that the user must take (ask a manager, connect VPN, run Dell Command Update, etc.).  

		   (You may omit the `"next"` field for free‑text answers.)

		4. **Limit output** – include only strictly required probes (≈5 or fewer).

		5. **Return JSON** in this exact schema and order:

		```json
		{
		  "probes": [
		    {
		      "question": "Do you have access to your Gilead email to receive the TruU enrollment code?",
		      "answer": {
		        "type": "yes_no",
		        "options": [
		          { "value": "Yes", "next": "proceed" },
		          { "value": "No",  "next": "Ask your manager or call the IT Service Desk to request a new TruU enrollment code, then return here." }
		        ]
		      }
		    },
		    {
		      "question": "What device and operating system are you using (Windows laptop, Mac, etc.)?",
		      "answer": {
		        "type": "one_of",
		        "options": [
		          { "value": "Windows laptop", "next": "proceed" },
		          { "value": "Mac",           "next": "proceed" },
		          { "value": "Other",         "next": "If you’re on a mobile device, follow the mobile‑specific article <link>; otherwise contact the Service Desk." }
		        ]
		      }
		    }
		    /* …more probes… */
		  ]
		}


		If there are no outstanding prerequisites, return:

		{ "probes": [] }
	
	'''

	user_prompt = f'''
		Analyze this technical document and generate probing questions. 
		==========================
		{document}
		==========================
	'''

	response = openai_client.chat.completions.create(
					model="gpt-4o",
					messages=[
						{"role": "system", "content": system_prompt},
						{"role": "user", "content": user_prompt}
						],
					temperature=0.1,
					response_format={"type": "json_object"}
				)

	extract_response = json.loads(response.choices[0].message.content)

	return extract_response